#!/usr/bin/env node
import { SIMDVectorEngine } from './simd/vector-wasm.js';

console.log(`
===========================================================
  ⚡ RUST/WASM SIMD VECTOR TENSOR COMPUTE CLI [v1.0.0]
  Author: anderdebona
===========================================================
`);

console.log('🚀 Running 10,000 iterations over 1,024-dimensional Float32 vectors...');

const result = SIMDVectorEngine.benchmarkSIMDVsScalar(1024, 10000);

console.log('\n📊 Benchmark Metrics:');
console.log(`• Scalar Loop Execution Time: ${result.scalarTimeMs.toFixed(2)} ms`);
console.log(`• SIMD Vectorized Execution Time: ${result.simdTimeMs.toFixed(2)} ms`);
console.log(`⚡ SIMD Speedup Factor: ${result.speedupFactor.toFixed(2)}x Throughput Acceleration`);
