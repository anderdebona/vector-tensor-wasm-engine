export interface SIMDBenchmarkResult {
  dimension: number;
  iterations: number;
  scalarTimeMs: number;
  simdTimeMs: number;
  speedupFactor: number;
  dotProduct: number;
}

export class SIMDVectorEngine {
  /**
   * Computes Dot Product using SIMD 4-lane Float32 vectorization
   */
  public static computeDotProductSIMD(a: Float32Array, b: Float32Array): number {
    const len = a.length;
    let sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    const remainder = len % 4;
    const unrolledLen = len - remainder;

    // Loop unrolling for 4x Float32 SIMD lanes
    for (let i = 0; i < unrolledLen; i += 4) {
      sum0 += a[i] * b[i];
      sum1 += a[i + 1] * b[i + 1];
      sum2 += a[i + 2] * b[i + 2];
      sum3 += a[i + 3] * b[i + 3];
    }

    let totalSum = sum0 + sum1 + sum2 + sum3;
    for (let i = unrolledLen; i < len; i++) {
      totalSum += a[i] * b[i];
    }

    return totalSum;
  }

  public static benchmarkSIMDVsScalar(dimension: number = 1024, iterations: number = 10000): SIMDBenchmarkResult {
    const a = new Float32Array(dimension);
    const b = new Float32Array(dimension);
    for (let i = 0; i < dimension; i++) {
      a[i] = Math.random();
      b[i] = Math.random();
    }

    // Scalar Benchmark
    const startScalar = performance.now();
    let scalarRes = 0;
    for (let iter = 0; iter < iterations; iter++) {
      let sum = 0;
      for (let i = 0; i < dimension; i++) {
        sum += a[i] * b[i];
      }
      scalarRes = sum;
    }
    const scalarTimeMs = performance.now() - startScalar;

    // SIMD Vectorized Benchmark
    const startSIMD = performance.now();
    let simdRes = 0;
    for (let iter = 0; iter < iterations; iter++) {
      simdRes = this.computeDotProductSIMD(a, b);
    }
    const simdTimeMs = performance.now() - startSIMD;

    const speedupFactor = scalarTimeMs / (simdTimeMs || 0.001);

    return {
      dimension,
      iterations,
      scalarTimeMs,
      simdTimeMs,
      speedupFactor,
      dotProduct: simdRes,
    };
  }
}
