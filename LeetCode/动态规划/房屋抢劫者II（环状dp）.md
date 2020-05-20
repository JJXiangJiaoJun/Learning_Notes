# 213. House Robber II

[题目链接](https://leetcode.com/problems/house-robber-ii/)

[TOC]

### 思路

* 类似于`3n分块的pizza`，第一个元素与最后一个元素是相邻的，那么这两个元素一定不能同时选择
* 我们将其分为两种情况：
    * 考虑`[0,1,2,...,n-1]`
    * 考虑`[1,2,3,...,n]`
* 环状结构上的DP都可以用上述方法解决



#### 动态规划

```cpp
class Solution {
public:
    int dp(vector<int>& nums,int last)
    {
        int len = nums.size();
        vector<int> d(len + 3,0);
        d[1] = nums[last];   
        for(int i = 2 ;i <= len - 1; i++)
        {
            d[i] = max(d[i-1],d[i-2] + nums[i - 1 + last]);
        }
        
        
        return d[len - 1];
    }
    
    int rob(vector<int>& nums) {
        int len = nums.size();
        if(len == 0) return 0;
        if(len == 1) return nums[0];
        
        return max(dp(nums,0),dp(nums,1));
        
    }
};
```

