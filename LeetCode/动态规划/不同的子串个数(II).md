# 940. Distinct Subsequences II

[题目链接](https://leetcode.com/problems/distinct-subsequences-ii/)

[TOC]

### 思路
* `dp[i]`表示，考虑`[0,i-1]`字母时总的方案数
    * `当s[i-1]为没有出现过的字符时`,`dp[i] = dp[i-1]*2 + 1`（原来的方案，加上最后以S[i-1]结尾的方案，加上只有S[i-1]一个数的方案）
    * `当S[i-1]为出现过的字符时`,`dp[i] = dp[i-1]*2 - 重复`，此时我们需要去重复，我们找到前一个`S[i-1]`字符的下标`j`，那么`dp[i] = dp[i-1]*2 - dp[j-1]`



### 代码

#### 动态规划

```cpp
class Solution {
public:
    typedef long long LL;
    int distinctSubseqII(string S) {
        int len = S.length();
        vector<LL> dp(len+10,0);
        dp[1]= 1;
        unordered_map<char,int> ch2idx;
        int m = 1e9+7; 
        
        for(int i=1;i<=len;i++)
        {
            if(!ch2idx.count(S[i-1]))
            {
                dp[i] = (dp[i-1]*2+1 +m)%m;
            }
            else
                dp[i] = (dp[i-1]*2 - dp[ch2idx[S[i-1]]-1]+m)%m;
            ch2idx[S[i-1]] = i;
        }
        
        return dp[len];
    }
};
```

