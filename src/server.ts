import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { SIMDVectorEngine } from './simd/vector-wasm.js';
import { SIMDMatrixEngine } from './simd/matrix-engine.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3009;

app.use(express.json());
app.use(express.static(path.join(__dirname, '../public')));

app.post('/api/simd/benchmark', (req, res) => {
  const result = SIMDVectorEngine.benchmarkSIMDVsScalar(1024, 10000);
  res.json(result);
});

app.post('/api/matrix/multiply', (req, res) => {
  const size = 64;
  const a = new Float32Array(size * size);
  const b = new Float32Array(size * size);
  for (let i = 0; i < size * size; i++) {
    a[i] = Math.random();
    b[i] = Math.random();
  }

  const result = SIMDMatrixEngine.multiplyMatrices(a, b, size);
  res.json(result);
});

app.listen(PORT, () => {
  console.log(`🚀 SIMD Vector Tensor Compute Engine running on http://localhost:${PORT}`);
});
