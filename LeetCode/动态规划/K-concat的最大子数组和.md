# 1191. K-Concatenation Maximum Sum

[题目链接](https://leetcode.com/problems/k-concatenation-maximum-sum/)

[TOC]

### 思路

![](https://assets.leetcode.com/users/dirkbe11/image_1568521165.png)

* 最大子数组和一共有三种情况：
    * 最大子数组和在中间，添加的数组没有任何影响
    * 最大子数组在中间，但是数组总和大于0，那么可以一直累加， arr = [-1, 4, -1], arr2 = [-1, **（4, -1, -1）**, 4 ,-1],括号括起来的地方就是数组的总和，因为大于0所以可以一直累加，`result = result + (k - 1) * sum`
    * 第三种情况就是最大子数组是在结尾，那么可以接上头部的数组。


#### 动态规划

```cpp
 class Solution {
public:
    typedef long long LL;
    int kConcatenationMaxSum(vector<int>& arr, int k) {
        LL mod = 1e9+7;
        LL ans = 0;
        LL sum = 0;
        LL cur_max = 0;
        for(int i = 0; i<arr.size();i++)
        {
            cur_max = max((LL)arr[i],arr[i] + cur_max);
            ans = max(ans,cur_max);
            sum  = (sum + arr[i])%mod;
        }
        
        
        
        if(k<2) return ans % mod;
        
        if(sum > 0) return ((k-1)*sum + ans) % mod;
        
        for(int i = 0; i < arr.size();i++)
        {
            cur_max = max((LL)arr[i],arr[i] + cur_max);
            ans = max(ans,cur_max);
        }
        
        return ans %mod;
    }
};
```

