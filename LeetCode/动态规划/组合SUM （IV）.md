[TOC]
# 377. Combination Sum IV

[题目链接](https://leetcode.com/problems/combination-sum-iv/)

[TOC]

### 思路
* `dp[i]`表示，`和为target`的时候的方案数。
* 每次都可以选择每个数字

### 代码

#### 动态规划

```cpp
class Solution {
public:
    typedef unsigned long long LL;
    int combinationSum4(vector<int>& nums, int target) {
      vector<LL> result(target +1);
        result[0] = 1;
        for(int i=1;i<=target;i++)
        {
            for(int &x:nums)
            {
                if(i>=x) result[i] += result[i-x];
            }
        }
        
        return result[target];
    }
};
```

