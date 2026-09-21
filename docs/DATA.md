# Data provenance

The original study downloaded adjusted-close prices through yfinance from Yahoo Finance for SPY, QQQ, AAPL, JPM, and XOM. The requested historical start was January 1, 2010; the frozen snapshot ends December 17, 2025. The paper and original repository are the provenance records; an independent download timestamp was not preserved.

`data_processed/manifest.json` records SHA-256 hashes of the existing public repository's prices and returns. The reproduction command verifies them and independently checks `log(prices.ffill()).diff()` against the stored returns. Checksums establish identity, not source accuracy or redistribution rights. No new data license is asserted by this portfolio release.

The original preprocessing forward-fills missing prices without a gap-length limit, despite its former “small gaps” wording. The verifier checks that this exact transformation reproduces the snapshot. This behavior should be reviewed before using the downloader on other instruments or periods.

For an optional refresh, install `requirements-download.txt` and run `python preprocess.py`. The downloader now fixes the exclusive end date at December 18, 2025 and writes into `reproduced/refreshed_data/`. Vendor revisions and adjusted-price changes can still alter historical values. Those files are not the paper's frozen inputs and do not replace them automatically.

The original PDF is copied byte-for-byte from the Semester 7 Advanced Econometrics folder. Its authors are **Lyla Saigal and Aaryaman Jaising**. It is included as the jointly authored course paper, without a new blanket license over coauthored material, third-party figures, or the data.
