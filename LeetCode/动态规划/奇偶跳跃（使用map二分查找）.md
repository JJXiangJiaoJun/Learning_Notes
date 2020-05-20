# 975. Odd Even Jump

[题目链接](https://leetcode.com/problems/odd-even-jump/)

[TOC]

### 思路
* 定义`dp[i][0]`为是否能在奇数步能够跳到最后一点，`dp[i][1]`为是否能在偶数步跳到最后一点
    * `dp[i][0] = dp[index_next_greater_number][1]` 因为下一步是偶数
    * `dp[i][1] = dp[index_next_smaller_number][0]`因为下一步是奇数
* 查找值可以用map，可以在`O(logN)`时间内找到




### 代码

#### 动态规划

```cpp
class Solution {
public:
    int oddEvenJumps(vector<int>& A) {
        int len = A.size();
        vector<vector<int>> dp(len+1,vector<int>(2,0));
        map<int,int>  num2idx;
        dp[len-1][0] = 1;
        dp[len-1][1] = 1;
        int ans = 1;
        num2idx[A[len-1]] = len-1;
        for(int i =  len - 2;i>=0;i--)
        {
            auto low = num2idx.lower_bound(A[i]);
            auto high = num2idx.upper_bound(A[i]);
            
            dp[i][0] =  dp[low->second][1];
            if(high!=num2idx.begin())
            {
                high--;
                dp[i][1] = dp[high->second][0];
            }
            
            ans += dp[i][0] ==1 ?1:0;
            num2idx[A[i]] = i;
        }
        
        return ans;
        
    }
};
```

