# Scaling Machine Learning Workflows by Moving from Python to Spark

This project demonstrates scaling machine learning workflows using Apache Spark.

## Business context

Python is slow. For data scientists or business analysts doing one experiment at a time, that slowness doesn't matter. And it is easy to trade some time for the benefit of easy use and plentiful analytics libraries like scikit-learn and pandas.

Apache Spark is a distributed computing framework designed for distributed data processing and ML. It allows you to write code once and run it anywhere --- on your laptop, a cloud cluster, or inside a managed platform like Databricks. Spark's MLlib package implements many of the common algorithms in sci-kit learn, but does so in a way designed for horizontal scalability (ie using more machines).

Cool idea. But if your code is all in Python, the fact that Spark is faster doesn't really help you.

## Article

Medium article: [Scaling Machine Learning Workflows by Moving from Python to Spark](https://medium.com/@kylejones_47003/scalingmlwithspark)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Spark ML functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Dataset size (samples, features)
- Spark configuration
- Output settings

## Spark ML Features

Spark capabilities:
- Distributed Processing: Handle large datasets
- MLlib: Machine learning library
- Vector Assembler: Feature preparation
- Scalable Algorithms: Logistic regression, etc.

## Caveats

- By default, generates synthetic large dataset.
- Full Spark implementation requires Spark cluster setup.
- PySpark requires Java runtime environment.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).