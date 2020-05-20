# 1278. Palindrome Partitioning III
[题目链接](https://leetcode.com/problems/palindrome-partitioning-iii/)

[TOC]
### 思路
* 二维DP，定义两个dp数组
    * `pal[i][j]`，表示把`i~j`子串变成回文串，需要修改的最少的字母数。
        * 当`s[i]==s[j]`时,`pal[i][j] = pal[i+1][j-1]`
        * 当`s[i]!=s[j]`时,`pal[i][j] = pal[i+1][j-1]+1`
    * `dp[i][j]`,表示`从0~i子串，分割为j个字符串时需要的最小修改次数`
        * `dp[i][j] = min(dp[l][j-1] + pal[l+1][i])  j-2<=l<i` 
* **注意以后这种有两个输入的都可以定义为两个输入的dp数组**，这里是`s长度和k`

### 代码

#### 动态规划(从后往前)

```cpp
class Solution {
public:
    int palindromePartition(string s, int k) {
        int len = s.length();
        vector<vector<int> >  pal(len,vector<int>(len,0));
        vector<vector<int> >  DP(len,vector<int>(k+2,0));
        
        for(int i=0;i<len-1;i++)
            if(s[i]==s[i+1]) pal[i][i+1] = 0;
            else pal[i][i+1] = 1;
        
        for(int t = 3;t<=len;t++)
            for(int i=0;i+t-1<len;i++)
            {
                int l = i,r = i+t-1;
                pal[l][r] = 1000;
                if(s[l]==s[r]) pal[l][r] = pal[l+1][r-1];
                else pal[l][r] = pal[l+1][r-1]+1;
            }
        
        for(int i=0;i<len;i++)
            DP[i][1] = pal[0][i];
        
        for(int l=2;l<=k;l++)
            for(int i=l-1;i<len;i++)
            {
                DP[i][l] = 10000;
                
                for(int m = l-2;m < i; m++)
                {
                    DP[i][l] = min(DP[i][l],DP[m][l-1]+pal[m+1][i]);
                }
            }
        
        return DP[len-1][k];
        
    }
};
```
