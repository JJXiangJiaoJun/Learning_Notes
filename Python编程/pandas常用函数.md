[TOC]
# 前言
* `pandas`库是一个科学数据处理库，常用来对数据进行预处理，数据分析等

# 读取数据
## `pd.read_csv`
```
pandas.read_csv(filepath_or_buffer, sep=', ', delimiter=None, header='infer', names=None, index_col=None, usecols=None, squeeze=False, prefix=None, mangle_dupe_cols=True, dtype=None, engine=None, converters=None, true_values=None, false_values=None, skipinitialspace=False, skiprows=None, nrows=None, na_values=None, keep_default_na=True, na_filter=True, verbose=False, skip_blank_lines=True, parse_dates=False, infer_datetime_format=False, keep_date_col=False, date_parser=None, dayfirst=False, iterator=False, chunksize=None, compression='infer', thousands=None, decimal=b'.', lineterminator=None, quotechar='"', quoting=0, escapechar=None, comment=None, encoding=None, dialect=None, tupleize_cols=None, error_bad_lines=True, warn_bad_lines=True, skipfooter=0, skip_footer=0, doublequote=True, delim_whitespace=False, as_recarray=None, compact_ints=None, use_unsigned=None, low_memory=True, buffer_lines=None, memory_map=False, float_precision=None)[source]

```
* 下面说下比较常用的参数
    * **`filepath_or_buffer `** :字符串，文件路径，或者文件句柄，或者字符串IO
    * **`sep`**: 字符串，分割符，默认值为‘，’
    * **`header`**：整数，或整数列表，缺省值  ‘infer’;指定第几行为表头
    * **`skiprows`**：值为整数或者可调用的函数，当指为整数时，作用是从文件头开始跳过无用的数据行(以0为起始下标)。当skiprows是一个可以调用的函数时，会读取符合该函数定义的规则的行。
    * **`names`**：列名数组，当没有表头时，可以用这个作为列名
    * **`error_bad_lines`**：设置为False可以跳过解析错误的行
