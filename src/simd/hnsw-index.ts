export interface HNSWNode { id: string; vector: Float32Array; neighbors: string[]; layer: number; }
export class HNSWIndex {
  private nodes: Map<string, HNSWNode> = new Map();
  private maxLayer: number = 0;
  private efConstruction: number;
  constructor(efConstruction: number = 16) { this.efConstruction = efConstruction; }
  private distance(a: Float32Array, b: Float32Array): number {
    let sum = 0; for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2; return Math.sqrt(sum);
  }
  public insert(id: string, vector: Float32Array): void {
    const layer = Math.floor(-Math.log(Math.random()) * 1.0);
    const node: HNSWNode = { id, vector, neighbors: [], layer };
    if (this.nodes.size > 0) {
      const allNodes = Array.from(this.nodes.values());
      const nearest = allNodes.sort((a, b) => this.distance(vector, a.vector) - this.distance(vector, b.vector)).slice(0, this.efConstruction);
      node.neighbors = nearest.map(n => n.id);
      nearest.forEach(n => { if (n.neighbors.length < this.efConstruction) n.neighbors.push(id); });
    }
    this.nodes.set(id, node);
    this.maxLayer = Math.max(this.maxLayer, layer);
  }
  public search(query: Float32Array, topK: number = 5): Array<{ id: string; distance: number }> {
    return Array.from(this.nodes.values())
      .map(n => ({ id: n.id, distance: this.distance(query, n.vector) }))
      .sort((a, b) => a.distance - b.distance).slice(0, topK);
  }
  public getStats(): { totalNodes: number; maxLayer: number } {
    return { totalNodes: this.nodes.size, maxLayer: this.maxLayer };
  }
}
