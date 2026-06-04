Run the data pipeline from the root/base dir
`-m` runs the module `pipeline.pipeline`
```bash
python -m pipeline.pipeline 2024
python -m pipeline.pipeline 2021 2022 2024 --mode viz
python -m pipeline.pipeline 2021-2024 --offline
```

```bash
python -m pytest
```

Run the test