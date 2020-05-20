# 378. 有序矩阵中第K小的元素
[TOC]

[题目链接](https://leetcode-cn.com/problems/kth-smallest-element-in-a-sorted-matrix/)

### 思路
* 用**二分查找**做，找出矩阵中最小元素left，最大元素right，那么答案一定在`[left,right]`中。
* 每次计算中点`mid`，然后计算矩阵中**小于等于**`mid`元素的个数cnt
    *  `if cnt<k`,那么答案一定在`[mid +1 ,right]`中，**`mid`不可能为答案**
    *  否则答案在 **`[left,mid]`** 中
### 代码

#### 
```cpp
class Solution {
public:
    int kthSmallest(vector<vector<int>>& matrix, int k) {
            int rows = matrix.size();
            int cols = matrix[0].size();

            int l = matrix[0][0],r = matrix.back().back();

            while(l<r)
            {
                int cnt = 0;
                int m = l + (r - l)/2;
                for(int i=0;i<rows;i++)
                     cnt += upper_bound(matrix[i].begin(),matrix[i].end(),m) - matrix[i].begin();
                if(cnt>=k) r = m;
                else l = m + 1;
            }

            return l;

    }
};
```

