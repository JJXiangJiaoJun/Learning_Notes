# 140. 单词拆分 II
[TOC]

[题目链接](https://leetcode-cn.com/problems/word-break-ii/)

### 思路
* 首先用动态规划判断，`dp[i]`表示前`i`个字符是否能拆分为字典里的词
* 然后回溯打印解即可
### 代码

#### 
```cpp
class Solution {
public:
    vector<string> ans;
    void dfs(string &s,int pos,string temp,unordered_set<string> &vis,vector<int> &dp)
    {
        if(pos<0) {ans.push_back(temp);return;}

        int original_size = temp.length();
        for(int i=pos;i>=0;i--)
        {
            string cur = s.substr(i,pos-i+1);
            if(vis.count(cur)&&dp[i])
            {
                if(temp.length()>0) temp = " " + temp;
                temp = cur + temp;

                dfs(s,i-1,temp,vis,dp);

                temp = temp.substr(temp.length() - original_size,original_size);
            }
        }

    }
    vector<string> wordBreak(string s, vector<string>& wordDict) {
        int len = s.length();
        vector<int> dp(len+1,0);
        dp[0] = 1;
        unordered_set<string> vis(wordDict.begin(),wordDict.end());
        for(int i=1;i<=len;i++)
        {
            for(int j=0;j<=i-1;j++)
            {
                if(vis.count(s.substr(j,i-j)))
                    dp[i] |= dp[j];
            }
        }

        string temp;

        dfs(s,s.length()-1,temp,vis,dp);

        return ans;

    }
};
```

