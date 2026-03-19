## Core Concepts
- `np.random.seed(42)`: **Reproducible randomness** - same seed = same sequence
- **Distributions**: 
  - `randint(1,7,size)` → integers [1,7)
  - `rand(n)` / `random()` → uniform [0,1)
  - `uniform(low,high,size)` → uniform [low,high)
  - `normal(mu,sigma,size)` → Gaussian bell curve
- **Array ops**: 
  - `rand(3,4)` → 2D uniform array
  - `choice(array, size, replace)` → sample from existing array
  - `shuffle(array)` → **in-place** permutation
  - `permutation(array)` → shuffled **copy**

## ML Use Cases
1. **Fake data**: `normal(5,2,1000)` + labels
2. **Weight init**: `normal(0,0.01,(input,output))` 
3. **Augmentation**: shuffle + `normal(0,0.05,shape)` noise

## Key Gotchas
- `shuffle()` modifies original array in-place
- Seed affects **all** subsequent random calls

