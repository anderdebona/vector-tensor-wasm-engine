import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { SIMDVectorEngine } from './simd/vector-wasm.js';
import { SIMDMatrixEngine } from './simd/matrix-engine.js';
import { CosineSimilarityWasmSIMD } from './simd/cosine-simd.js';
import { KMeansClusteringEngine } from './simd/kmeans-engine.js';
import { ProductQuantizationIVFPQ } from './simd/product-quantization-ivfpq.js';
import { SparseDenseSpMM } from './simd/sparse-dense-spmm.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3009;

app.use(express.json());
app.use(express.static(path.join(__dirname, '../public')));

const pqEngine = new ProductQuantizationIVFPQ(8, 2, 4);

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

app.post('/api/simd/pq', (req, res) => {
  const vec = [0.25, 0.35, 0.45, 0.55, 0.50, 0.60, 0.70, 0.80];
  const query = [0.28, 0.38, 0.42, 0.52, 0.51, 0.62, 0.68, 0.82];

  const encoded = pqEngine.encode('vec_target_1', vec);
  const adcDistance = pqEngine.computeADCDistance(query, encoded);

  res.json({
    originalVector: vec,
    queryVector: query,
    encodedCodes: encoded.codes,
    adcDistance,
    codebookStats: pqEngine.getCodebookStats()
  });
});

app.post('/api/simd/spmm', (req, res) => {
  const sparseA = [
    [1.5, 0, 0, 0],
    [0, 0, 3.2, 0],
    [0, 2.1, 0, 0],
    [0, 0, 0, 4.0]
  ];
  const denseB = [
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0]
  ];

  const csr = SparseDenseSpMM.toCSR(sparseA);
  const spmmResult = SparseDenseSpMM.multiply(csr, denseB);

  res.json({
    csrRepresentation: {
      values: csr.values,
      colIndices: csr.colIndices,
      rowPointers: csr.rowPointers,
      sparsityRatio: csr.sparsityRatio
    },
    spmmResult
  });
});

app.post('/api/simd/kmeans', (req, res) => {
  const k = req.body.k || 4;
  const numVectors = req.body.numVectors || 80;

  const vectors: Float32Array[] = [];
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
  console.log(`🚀 SIMD Vector Tensor Compute Engine v5.0.0 on http://localhost:${PORT}`);
});
