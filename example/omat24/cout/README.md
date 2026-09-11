# OMat24 mix batch-size distribution

- Source: `/data/public/wuxingxing/metadata/decompress/Omat24/train`
- Valid ASE-LMDB shards: 5216
- Skipped invalid shards: 0
- Frames: 100,824,585
- Atoms: 1,883,602,758
- Average atoms/frame: 18.681979
- Minimum/maximum atoms/frame: 1 / 236
- Ordering: MatPL `BlockShuffleIndices`, seed=2023, epoch=0, block_size=65536
- Packing: identical greedy rule to `DistributedAtomBatchSampler._global_batches`
- Frequency: `count / total batches` for each mix value

Each `batchsize_frequency_mix_N.csv` stores the plotting data. Each mix value has an independent PNG, and `batchsize_distributions_all_mix.png` is the 2×5 overview.
