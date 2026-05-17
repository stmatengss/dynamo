// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mooncake Master health probe and circuit breaker for the KV Router.
//!
//! The [`MooncakeMasterHealthProbe`] runs a background polling loop that
//! periodically hits the Mooncake master metrics endpoint.  Results feed a
//! [`CircuitBreaker`] so that callers (e.g. `HicacheSharedKvCache`) can
//! short-circuit requests when the master is known to be unavailable, avoiding
//! the per-request HTTP timeout penalty.
//!
//! # Circuit breaker states
//!
//! ```text
//!  ┌────────┐  failure_threshold   ┌──────┐  recovery_timeout  ┌──────────┐
//!  │ Closed ├─────── exceeded ────►│ Open ├───── elapsed ────►│ HalfOpen │
//!  └───▲────┘                      └──────┘                   └────┬─────┘
//!      │         success                                           │
//!      └───────────────────────────────────────────────────────────┘
//!                                failure ──► back to Open
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use prometheus::{Histogram, HistogramOpts, IntCounter, IntGauge, Opts, Registry};
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Circuit breaker
// ---------------------------------------------------------------------------

/// Possible states for the circuit breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Healthy -- requests flow normally.
    Closed,
    /// Unhealthy -- requests are short-circuited.
    Open,
    /// Testing recovery -- a single probe is allowed through.
    HalfOpen,
}

/// A simple consecutive-failure circuit breaker.
///
/// Thread-safe access is provided by wrapping this struct in an
/// `Arc<RwLock<..>>`.
#[derive(Debug)]
pub struct CircuitBreaker {
    state: CircuitState,
    consecutive_failures: u32,
    failure_threshold: u32,
    recovery_timeout: Duration,
    last_failure_time: Option<Instant>,
    total_successes: u64,
    total_failures: u64,
}

impl CircuitBreaker {
    /// Create a new circuit breaker in the [`CircuitState::Closed`] state.
    pub fn new(failure_threshold: u32, recovery_timeout: Duration) -> Self {
        Self {
            state: CircuitState::Closed,
            consecutive_failures: 0,
            failure_threshold,
            recovery_timeout,
            last_failure_time: None,
            total_successes: 0,
            total_failures: 0,
        }
    }

    /// Returns `true` when requests should be attempted.
    ///
    /// - [`CircuitState::Closed`]: always available.
    /// - [`CircuitState::Open`]: available only after `recovery_timeout` has
    ///   elapsed since the last failure (transitions to [`CircuitState::HalfOpen`]).
    /// - [`CircuitState::HalfOpen`]: not available (one probe is already in
    ///   flight).
    pub fn is_available(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() >= self.recovery_timeout {
                        self.state = CircuitState::HalfOpen;
                        tracing::info!("Circuit breaker transitioning from Open to HalfOpen");
                        true
                    } else {
                        false
                    }
                } else {
                    // Should not happen, but be safe.
                    false
                }
            }
            CircuitState::HalfOpen => {
                // Only one probe at a time while half-open.
                false
            }
        }
    }

    /// Record a successful probe.  Resets the failure count and closes the
    /// circuit.
    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
        self.total_successes += 1;
        if self.state != CircuitState::Closed {
            tracing::info!(
                prev_state = ?self.state,
                "Circuit breaker closing after successful probe"
            );
            self.state = CircuitState::Closed;
        }
    }

    /// Record a failed probe.  Increments the failure counter and may open
    /// the circuit.
    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        self.total_failures += 1;
        self.last_failure_time = Some(Instant::now());

        if self.consecutive_failures >= self.failure_threshold {
            if self.state != CircuitState::Open {
                tracing::warn!(
                    consecutive_failures = self.consecutive_failures,
                    threshold = self.failure_threshold,
                    prev_state = ?self.state,
                    "Circuit breaker opening due to consecutive failures"
                );
            }
            self.state = CircuitState::Open;
        }
    }

    /// Current state of the circuit.
    pub fn state(&self) -> CircuitState {
        self.state
    }

    /// Number of consecutive failures since the last success.
    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures
    }

    /// Return a health score between 0.0 (completely unhealthy) and 1.0
    /// (fully healthy) based on the circuit state and failure history.
    pub fn health_score(&self) -> f64 {
        match self.state {
            CircuitState::Closed => {
                if self.failure_threshold == 0 {
                    return 1.0;
                }
                // Linearly degrade as failures approach the threshold.
                let ratio = self.consecutive_failures as f64 / self.failure_threshold as f64;
                (1.0 - ratio).max(0.0)
            }
            CircuitState::HalfOpen => 0.25,
            CircuitState::Open => 0.0,
        }
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(3, Duration::from_secs(30))
    }
}

// ---------------------------------------------------------------------------
// Prometheus metrics
// ---------------------------------------------------------------------------

/// Prometheus metrics emitted by the health probe.
struct HealthProbeMetrics {
    healthy: IntGauge,
    latency_seconds: Histogram,
    probe_failures_total: IntCounter,
    pool_utilization: IntGauge,
    cache_hit_rate: IntGauge,
}

impl HealthProbeMetrics {
    fn new(registry: &Registry) -> Self {
        let healthy = IntGauge::with_opts(
            Opts::new(
                "mooncake_master_healthy",
                "Whether the Mooncake master is reachable (1 = healthy, 0 = unhealthy)",
            ),
        )
        .expect("failed to create mooncake_master_healthy gauge");

        let latency_seconds = Histogram::with_opts(
            HistogramOpts::new(
                "mooncake_master_latency_seconds",
                "Latency of health probe requests to the Mooncake master",
            )
            .buckets(vec![0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]),
        )
        .expect("failed to create mooncake_master_latency_seconds histogram");

        let probe_failures_total = IntCounter::with_opts(
            Opts::new(
                "mooncake_probe_failures_total",
                "Total number of failed health probe requests to the Mooncake master",
            ),
        )
        .expect("failed to create mooncake_probe_failures_total counter");

        let pool_utilization = IntGauge::with_opts(
            Opts::new(
                "mooncake_pool_utilization",
                "Mooncake object pool utilization as reported by the master (percentage 0-100)",
            ),
        )
        .expect("failed to create mooncake_pool_utilization gauge");

        let cache_hit_rate = IntGauge::with_opts(
            Opts::new(
                "mooncake_cache_hit_rate",
                "Mooncake cache hit rate as reported by the master (percentage 0-100)",
            ),
        )
        .expect("failed to create mooncake_cache_hit_rate gauge");

        registry
            .register(Box::new(healthy.clone()))
            .expect("failed to register mooncake_master_healthy");
        registry
            .register(Box::new(latency_seconds.clone()))
            .expect("failed to register mooncake_master_latency_seconds");
        registry
            .register(Box::new(probe_failures_total.clone()))
            .expect("failed to register mooncake_probe_failures_total");
        registry
            .register(Box::new(pool_utilization.clone()))
            .expect("failed to register mooncake_pool_utilization");
        registry
            .register(Box::new(cache_hit_rate.clone()))
            .expect("failed to register mooncake_cache_hit_rate");

        Self {
            healthy,
            latency_seconds,
            probe_failures_total,
            pool_utilization,
            cache_hit_rate,
        }
    }
}

// ---------------------------------------------------------------------------
// Health probe
// ---------------------------------------------------------------------------

/// Background health probe for the Mooncake master HTTP service.
///
/// Spawns a polling task that periodically issues a lightweight HTTP request
/// to the master metrics endpoint and feeds the result into a circuit breaker.
/// Consumers call [`is_healthy`](Self::is_healthy) or
/// [`get_health_score`](Self::get_health_score) for fast, lock-free-ish
/// checks without incurring their own HTTP round-trip.
pub struct MooncakeMasterHealthProbe {
    metrics_url: String,
    client: reqwest::Client,
    circuit_breaker: Arc<RwLock<CircuitBreaker>>,
    poll_interval: Duration,
    metrics: Arc<HealthProbeMetrics>,
}

/// Lightweight response we expect from the Mooncake master metrics endpoint.
/// Fields are optional because the exact schema may evolve.
#[derive(Debug, serde::Deserialize)]
struct MooncakeMetricsResponse {
    #[serde(default)]
    pool_utilization: Option<f64>,
    #[serde(default)]
    cache_hit_rate: Option<f64>,
}

/// Short HTTP timeout for health probes -- we want to detect outages fast.
const HEALTH_PROBE_TIMEOUT: Duration = Duration::from_secs(3);

impl MooncakeMasterHealthProbe {
    /// Create a new health probe.
    ///
    /// * `metrics_url`  -- Full URL to the Mooncake master metrics endpoint
    ///   (e.g. `http://mooncake-master:9003/metrics`).
    /// * `poll_interval` -- How often to poll (default recommendation: 10 s).
    /// * `registry` -- Prometheus registry to register health metrics on.
    pub fn new(
        metrics_url: String,
        poll_interval: Duration,
        registry: &Registry,
    ) -> Self {
        let client = reqwest::Client::builder()
            .timeout(HEALTH_PROBE_TIMEOUT)
            .build()
            .expect("failed to build health-probe reqwest client");

        Self {
            metrics_url,
            client,
            circuit_breaker: Arc::new(RwLock::new(CircuitBreaker::default())),
            poll_interval,
            metrics: Arc::new(HealthProbeMetrics::new(registry)),
        }
    }

    /// Spawn the background polling loop.
    ///
    /// The task runs until `cancel_token` is cancelled and then exits
    /// gracefully.  The returned [`JoinHandle`] can be used to await
    /// completion.
    pub fn spawn_probe(&self, cancel_token: CancellationToken) -> JoinHandle<()> {
        let client = self.client.clone();
        let url = self.metrics_url.clone();
        let cb = Arc::clone(&self.circuit_breaker);
        let interval = self.poll_interval;
        let metrics = Arc::clone(&self.metrics);

        tokio::spawn(async move {
            tracing::info!(
                url = %url,
                poll_interval_secs = interval.as_secs(),
                "Mooncake master health probe started"
            );

            loop {
                tokio::select! {
                    _ = cancel_token.cancelled() => {
                        tracing::info!("Mooncake master health probe shutting down");
                        break;
                    }
                    _ = tokio::time::sleep(interval) => {
                        Self::probe_once(&client, &url, &cb, &metrics).await;
                    }
                }
            }
        })
    }

    /// Execute a single probe iteration.
    async fn probe_once(
        client: &reqwest::Client,
        url: &str,
        cb: &Arc<RwLock<CircuitBreaker>>,
        metrics: &HealthProbeMetrics,
    ) {
        let start = Instant::now();
        let result = client.get(url).send().await;
        let elapsed = start.elapsed();

        metrics.latency_seconds.observe(elapsed.as_secs_f64());

        match result {
            Ok(response) if response.status().is_success() => {
                // Try to parse optional metrics from the response body.
                if let Ok(body) = response.json::<MooncakeMetricsResponse>().await {
                    if let Some(pool_util) = body.pool_utilization {
                        metrics
                            .pool_utilization
                            .set((pool_util * 100.0).round() as i64);
                    }
                    if let Some(hit_rate) = body.cache_hit_rate {
                        metrics
                            .cache_hit_rate
                            .set((hit_rate * 100.0).round() as i64);
                    }
                }

                metrics.healthy.set(1);
                let mut breaker = cb.write().await;
                breaker.record_success();

                tracing::trace!(
                    latency_ms = elapsed.as_millis(),
                    "Mooncake master health probe succeeded"
                );
            }
            Ok(response) => {
                let status = response.status();
                metrics.healthy.set(0);
                metrics.probe_failures_total.inc();
                let mut breaker = cb.write().await;
                breaker.record_failure();

                tracing::warn!(
                    status = %status,
                    latency_ms = elapsed.as_millis(),
                    consecutive_failures = breaker.consecutive_failures(),
                    "Mooncake master health probe returned non-success status"
                );
            }
            Err(error) => {
                metrics.healthy.set(0);
                metrics.probe_failures_total.inc();
                let mut breaker = cb.write().await;
                breaker.record_failure();

                tracing::warn!(
                    %error,
                    latency_ms = elapsed.as_millis(),
                    consecutive_failures = breaker.consecutive_failures(),
                    "Mooncake master health probe request failed"
                );
            }
        }
    }

    /// Fast check: is the Mooncake master considered healthy?
    ///
    /// This acquires a write lock briefly because `is_available` may
    /// transition from `Open` to `HalfOpen`.  The lock is uncontended in
    /// the common (healthy) case and is held for a trivial duration.
    pub async fn is_healthy(&self) -> bool {
        let mut breaker = self.circuit_breaker.write().await;
        breaker.is_available()
    }

    /// Return a health score in `[0.0, 1.0]` suitable for weighting routing
    /// decisions.
    ///
    /// - 1.0 = fully healthy (circuit closed, zero recent failures)
    /// - 0.0 = completely unhealthy (circuit open)
    pub async fn get_health_score(&self) -> f64 {
        let breaker = self.circuit_breaker.read().await;
        breaker.health_score()
    }

    /// Obtain a reference to the underlying circuit breaker for advanced
    /// use cases (e.g. integration with `HicacheSharedKvCache`).
    pub fn circuit_breaker(&self) -> &Arc<RwLock<CircuitBreaker>> {
        &self.circuit_breaker
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- CircuitBreaker unit tests ------------------------------------------

    #[test]
    fn test_circuit_breaker_state_transitions() {
        let mut cb = CircuitBreaker::new(3, Duration::from_millis(50));

        // Starts closed.
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.is_available());

        // Record two failures -- still closed (threshold is 3).
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.is_available());

        // Third failure opens the circuit.
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.is_available());

        // Wait for recovery timeout to elapse.
        std::thread::sleep(Duration::from_millis(60));

        // Now `is_available` should transition to HalfOpen.
        assert!(cb.is_available());
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // While half-open, further calls return false (only one probe allowed).
        assert!(!cb.is_available());

        // A success closes the circuit.
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.is_available());
    }

    #[test]
    fn test_health_score_degradation() {
        let mut cb = CircuitBreaker::new(4, Duration::from_secs(30));

        // Fully healthy.
        assert!((cb.health_score() - 1.0).abs() < f64::EPSILON);

        // One failure -> 0.75
        cb.record_failure();
        assert!((cb.health_score() - 0.75).abs() < f64::EPSILON);

        // Two failures -> 0.50
        cb.record_failure();
        assert!((cb.health_score() - 0.50).abs() < f64::EPSILON);

        // Three failures -> 0.25
        cb.record_failure();
        assert!((cb.health_score() - 0.25).abs() < f64::EPSILON);

        // Fourth failure opens the circuit -> score 0.0
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!((cb.health_score() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_circuit_breaker_recovery() {
        let mut cb = CircuitBreaker::new(2, Duration::from_millis(30));

        // Open the circuit.
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        // Before timeout: still open.
        assert!(!cb.is_available());

        // Wait for recovery timeout.
        std::thread::sleep(Duration::from_millis(40));

        // Transitions to half-open.
        assert!(cb.is_available());
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // Another failure re-opens.
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        // Wait again, transition to half-open, then succeed.
        std::thread::sleep(Duration::from_millis(40));
        assert!(cb.is_available());
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.consecutive_failures(), 0);
    }

    #[test]
    fn test_circuit_breaker_success_resets_failures() {
        let mut cb = CircuitBreaker::new(5, Duration::from_secs(30));

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.consecutive_failures(), 2);

        cb.record_success();
        assert_eq!(cb.consecutive_failures(), 0);
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_default_circuit_breaker() {
        let cb = CircuitBreaker::default();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.failure_threshold, 3);
        assert_eq!(cb.recovery_timeout, Duration::from_secs(30));
    }

    // -- MooncakeMasterHealthProbe unit tests --------------------------------

    #[tokio::test]
    async fn test_probe_marks_healthy_on_success() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/metrics")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"pool_utilization": 0.42, "cache_hit_rate": 0.87}"#)
            .create_async()
            .await;

        let registry = Registry::new();
        let probe = MooncakeMasterHealthProbe::new(
            format!("{}/metrics", server.url()),
            Duration::from_secs(60),
            &registry,
        );

        // Simulate one probe.
        MooncakeMasterHealthProbe::probe_once(
            &probe.client,
            &probe.metrics_url,
            &probe.circuit_breaker,
            &probe.metrics,
        )
        .await;

        assert!(probe.is_healthy().await);
        assert!((probe.get_health_score().await - 1.0).abs() < f64::EPSILON);

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_probe_marks_unhealthy_on_failure() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/metrics")
            .with_status(500)
            .expect(3)
            .create_async()
            .await;

        let registry = Registry::new();
        let probe = MooncakeMasterHealthProbe::new(
            format!("{}/metrics", server.url()),
            Duration::from_secs(60),
            &registry,
        );

        // Three failures should open the circuit (default threshold = 3).
        for _ in 0..3 {
            MooncakeMasterHealthProbe::probe_once(
                &probe.client,
                &probe.metrics_url,
                &probe.circuit_breaker,
                &probe.metrics,
            )
            .await;
        }

        assert!(!probe.is_healthy().await);
        assert!((probe.get_health_score().await - 0.0).abs() < f64::EPSILON);

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_spawn_probe_respects_cancellation() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("GET", "/metrics")
            .with_status(200)
            .with_body("{}")
            .expect_at_least(0)
            .create_async()
            .await;

        let registry = Registry::new();
        let probe = MooncakeMasterHealthProbe::new(
            format!("{}/metrics", server.url()),
            Duration::from_millis(50),
            &registry,
        );

        let cancel = CancellationToken::new();
        let handle = probe.spawn_probe(cancel.clone());

        // Let the probe run a few iterations.
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Cancel and ensure the task finishes promptly.
        cancel.cancel();
        let result = tokio::time::timeout(Duration::from_secs(2), handle).await;
        assert!(result.is_ok(), "probe task should finish after cancellation");
    }
}
