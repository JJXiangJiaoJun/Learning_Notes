# 673. Number of Longest Increasing Subsequence

[题目链接](https://leetcode.com/problems/number-of-longest-increasing-subsequence/)

[TOC]

### 思路

* 用`vector<pair<int>> dp[i]`表示，考虑以`i`结尾的LIS长度和个数




#### 动态规划

```cpp
class Solution {
public:
    int findNumberOfLIS(vector<int>& nums) {
        int len = nums.size();
        if(len==0) return 0;
        vector<pair<int,int>> dp(len);
        dp[0].first= 1;
        dp[0].second = 1;
        int max_len = 1;
        for(int i = 1 ;i < len ;i++)
        {
            dp[i].first = 1;
            dp[i].second = 1;
            for(int j = 0 ;j < i;j++)
                if(nums[i]>nums[j])
                {
                    if(dp[i].first > dp[j].first + 1) continue;
                    if(dp[i].first == dp[j].first + 1) dp[i].second += dp[j].second;
                    else
                    {
                        dp[i].first = dp[j].first + 1;
                        dp[i].second = dp[j].second;
                    }
                }
            max_len = max(max_len,dp[i].first);
        }
        
        int ans = 0;
        for(auto &c:dp)
        {
            ans += max_len == c.first ? c.second : 0;
        }
        
        return ans;
            
        
    }
};
```

