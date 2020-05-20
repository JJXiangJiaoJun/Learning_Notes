[TOC]
# 698. Partition to K Equal Sum Subsets

[题目链接](https://leetcode.com/problems/partition-to-k-equal-sum-subsets/)

[TOC]

### 思路
* **用bitmask做，bitmask一般可以采用刷表法**
* `dp[i]`表示选取元素集合为`i`时，`sum % target`的值
* `dp[mask|(1<<i)] = (dp[mask]+nums[i]) % tar;`
* **这种元素选取的问题，都可以考虑用bitmask解决**

### 代码

#### 动态规划

```cpp
class Solution {
public:
    int dp[(1<<16)+10];
    bool canPartitionKSubsets(vector<int>& nums, int k) {
        int len = nums.size();
        int sum = 0;
        for(int i=0;i<len;i++)
            sum += nums[i];
        if(sum%k!=0) return false;
        
        memset(dp,-1,sizeof(dp));
        dp[0] = 0;
        int target = sum/k;
        
        for(int i=0;i<(1<<len);i++)
        {
            if(dp[i]==-1) continue;
            
            for(int j = 0;j<len;j++)
            {
                if((i&(1<<j))==0&&dp[i] + nums[j]<=target)
                {
                    dp[i|(1<<j)] = (dp[i] + nums[j])%target;
                }
                
            }
        }
        
        return dp[(1<<len) -1]==0;
    }
};
```

