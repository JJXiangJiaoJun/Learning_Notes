# 560. Subarray Sum Equals K
[题目链接](https://leetcode.com/problems/subarray-sum-equals-k/)

### 思路
* 枚举起点+终点，进行判断。使用前缀和在`O(1)`时间算出sum，总时间复杂度`O(n^2)`
* 使用`unordered_map<int,int>`，`map`为sum2cnt，即前缀中为sum的项的个数，那么只需枚举终点，然后每次查找`sum2cnt[sum-k]`，即为以当前元素为终点时，和为k的连续子数组的个数。(sum[i]-sum[j]=k,则`j ---> i`的和为k)
* !!!!注意，以后配上这种情况可以使用map等数据结构来优化时间。相当于枚举终点，然后在`O(1)`时间下找到起点。

### 代码

#### 枚举起点+终点

```cpp
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        vector<long long> cusum(nums.size()+2,0);
        for(int i=1;i<=nums.size();i++)
            cusum[i] =cusum[i-1] + nums[i-1];
        long long target = k;
        int ans=0;
        for(int i=0;i<nums.size();i++)
            for(int j=i;j<nums.size();j++)
            {
                long long cur_sum = cusum[j+1] - cusum[i];
                if(cur_sum==target) ans++;
            }
        return ans;
        
    }
};
```

#### 枚举终点+map优化

```cpp
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int ans=0,sum=0;
        unordered_map<int,int> sum2cnt;
        sum2cnt[0] = 1;
        for(int i=0;i<nums.size();i++)
        {
            sum+=nums[i];
            if(sum2cnt.count(sum-k))
            {
                ans+=sum2cnt[sum-k];
            }
            sum2cnt[sum] = sum2cnt.count(sum)?sum2cnt[sum]+1:1;
        }
        return ans;
    }
};
```