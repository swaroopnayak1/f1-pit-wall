
## Running the script

Run the data pipeline from the root/base dir
`-m` runs the module `pipeline.pipeline`
```bash
python -m pipeline.pipeline 2024
python -m pipeline.pipeline 2021 2022 2024 --mode viz
python -m pipeline.pipeline 2021-2024 --offline
```



## Run the test

How to run tests manually

```bash
python -m pytest
```

### Run all tests with verbose output
`& "C:\Users\SwaroopPC\anaconda3\envs\f1-pit-wall\Scripts\pytest.exe" tests\ -v`
### Run a single test file
`& "C:\Users\SwaroopPC\anaconda3\envs\f1-pit-wall\Scripts\pytest.exe" tests\test_registry.py -v`
### Run a specific test class or test
`& "C:\Users\SwaroopPC\anaconda3\envs\f1-pit-wall\Scripts\pytest.exe" tests\test_registry.py::TestRegister -v`
### Stop on first failure
`& "C:\Users\SwaroopPC\anaconda3\envs\f1-pit-wall\Scripts\pytest.exe" tests\ -x`
### Show only failures (no verbose)
`& "C:\Users\SwaroopPC\anaconda3\envs\f1-pit-wall\Scripts\pytest.exe" tests\`

How to read results
. or PASSED — test passed
F or FAILED — assertion failed, stacktrace shown below
E or ERROR — setup/teardown error (fixture problem)
Final line shows N passed / M failed in Xs — that's the summary to check