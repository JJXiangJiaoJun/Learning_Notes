[TOC]
# `os`模块的功能
* 编写Python脚本处理数据的过程中，经常需要对**文件查找，以及路径操作**，这就依赖于os模块

# 目录相关
## `os.getcwd()`
* 获取代码当前路径

## `os.listdir(path)`
* 列出当前目录下所有文件或文件夹

## `os.makedirs(path,mode=0o777)`
* 递归创建目录

```python
os.makedirs(path, mode=0o777)
```
## `os.chdir(path)`
* 改变当前工作目录

## `os.removedirs(path)`
* 用于递归删除目录，如果子文件夹成功删除, removedirs()才尝试它们的父文件夹,直到抛出一个error(它基本上被忽略,因为它一般意味着你文件夹不为空)。

## `os.rmdir(path)`
* os.rmdir() 方法用于删除指定路径的目录。仅当这文件夹是空的才可以, 否则, 抛出OSError。


# 文件操作相关
## `os.chown(path, uid, gid)`
* 更改文件的所有者

## `os.rename(src, dst)`
* 重命名文件或目录，从 src 到 dst

## `os.remove(path)`
* 删除路径为path的文件。如果path是一个文件夹，将抛出OSError; 查看下面的rmdir()删除一个 directory。


# `os.path`常用模块
## `os.path.abspath(path)`
* 返回绝对路径

## `os.path.basename(path)`
* 返回文件名

## `os.path.dirname(path)`
* 返回文件目录

## `os.path.exists(path)`
* 如果路径 path 存在，返回 True；如果路径 path 不存在，返回 False。

## `os.path.isabs(path)`
* 判断是否为绝对路径

## `os.path.isfile(path)`
* 判断路径是否为文件

## `os.path.isdir(path)`
* 判断路径是否为目录

## `os.path.join(path1[, path2[, ...]])`
* 把目录和文件名合成一个路径

## `os.path.split(path)`
* 把路径分割成 dirname 和 basename，返回一个元组