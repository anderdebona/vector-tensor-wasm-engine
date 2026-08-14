import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { SIMDVectorEngine } from './simd/vector-wasm.js';
import { SIMDMatrixEngine } from './simd/matrix-engine.js';
import { CosineSimilarityWasmSIMD } from './simd/cosine-simd.js';
import { KMeansClusteringEngine } from './simd/kmeans-engine.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3009;

app.use(express.json());
app.use(express.static(path.join(__dirname, '../public')));

app.post('/api/simd/benchmark', (req, res) => {
  const dimension = req.body.dimension || 1024;
  const iterations = req.body.iterations || 10000;
  const result = SIMDVectorEngine.benchmarkSIMDVsScalar(dimension, iterations);
  res.json(result);
});

app.post('/api/matrix/multiply', (req, res) => {
  const size = req.body.size || 64;
  const a = new Float32Array(size * size);
  const b = new Float32Array(size * size);
  for (let i = 0; i < size * size; i++) {
    a[i] = Math.random();
    b[i] = Math.random();
  }

  const result = SIMDMatrixEngine.multiplyMatrices(a, b, size);
  res.json(result);
});

app.post('/api/simd/kmeans', (req, res) => {
  const k = req.body.k || 4;
  const numVectors = req.body.numVectors || 80;
  const dim = 2; // 2D for direct canvas projection

  const vectors: Float32Array[] = [];
  // Generate k clusters around random centers
  const centerOffsets = Array.from({ length: k }, () => [Math.random() * 80 + 10, Math.random() * 80 + 10]);

  for (let i = 0; i < numVectors; i++) {
    const cluster = i % k;
    const center = centerOffsets[cluster];
    const v = new Float32Array([
      center[0] + (Math.random() - 0.5) * 15,
      center[1] + (Math.random() - 0.5) * 15,
    ]);
    vectors.push(v);
  }

  const engine = new KMeansClusteringEngine(k, 25);
  const fitResult = engine.fit(vectors);

  const points = vectors.map((v, idx) => ({
    x: v[0],
    y: v[1],
    cluster: fitResult.assignments[idx],
  }));

  const centroids = fitResult.centroids.map(c => ({
    x: c[0],
    y: c[1],
  }));

  res.json({
    k,
    iterations: fitResult.iterations,
    inertia: fitResult.inertia,
    points,
    centroids,
  });
});

app.post('/api/simd/cosine', (req, res) => {
  const query = new Float32Array([1.0, 0.8, 0.5, 0.2]);
  const candidates = [
    { id: 'doc-alpha', vector: new Float32Array([0.95, 0.82, 0.48, 0.21]) },
    { id: 'doc-beta', vector: new Float32Array([0.1, 0.9, 0.2, 0.8]) },
    { id: 'doc-gamma', vector: new Float32Array([0.88, 0.75, 0.40, 0.15]) },
    { id: 'doc-delta', vector: new Float32Array([-0.5, -0.2, 0.1, 0.0]) },
  ];

  const ranked = CosineSimilarityWasmSIMD.batchSearch(query, candidates, 4);
  res.json({ query: Array.from(query), ranked });
});

app.listen(PORT, () => {
  console.log(`🚀 SIMD Vector Tensor Compute Engine Turbocharged on http://localhost:${PORT}`);
});
