# Runtime & Space Complexity Analysis
**Moving Average Trading Strategy Optimization**  
*O(n²) → O(1) time/space progression across 5 implementations*

## 🎯 **Project Overview**
Demonstrates algorithm optimization by implementing 5 moving average strategies with **progressively better complexity**:

| Strategy | Time per tick | Space | Description |
|----------|---------------|-------|-------------|
| **Naive** | **O(n)** | **O(n)** | Full history, recompute mean each tick |
| **Windowed** | **O(1)** | **O(k)** | Fixed window + running sum |
| **Deque** | **O(k)** | **O(k)** | \`deque(maxlen=k)\` + mean() |
| **NumPy** | **O(1)** | **O(k)** | Circular buffer + vectorization |
| **Streaming** | **O(1)** | **O(1)** | EMA (no price storage) |

## 📦 **Setup**

\`\`\`bash
# Clone & install
git clone <your-repo>
cd runtime_and_space_complexity
pip install -r requirements.txt

# Generate test data (if no market_data.csv)
python generate_sample_data.py  # Optional: creates 126k ticks
\`\`\`

### **requirements.txt**
\`\`\`
numpy
matplotlib
pandas
tqdm
pytest
\`\`\`

## 🚀 **Usage**

\`\`\`bash
# Full analysis pipeline
python main.py
\`\`\`

**Outputs:**
- \`complexity_plots.png\` - Log-scale runtime/memory graphs
- \`complexity_report.md\` - Tables + theoretical analysis
- Console: cProfile hotspots + benchmark table

### **Expected Results (100K ticks)**
\`\`\`
=== 100,000 ticks ===
  Naive     : 423.1s 234.5MB  ← Quadratic disaster!
  Streaming :  0.01s   0.08MB  ← O(1) perfection
\`\`\`
