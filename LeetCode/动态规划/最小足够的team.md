[TOC]
# 1125. Smallest Sufficient Team

[题目链接](https://leetcode.com/problems/smallest-sufficient-team/)

[TOC]



### 思路
* 这种问题用**集合上的动态规划解决**，`dp[i]`表示能力集合为`i`时，**最小的满足team**，用`unordered_map<int,vector<int>>`保存结果
* 采用**刷表法**，每次增加当前这个人时，如果当前集合没有或者人数小于当前人数，那么就进行更新




### 代码

#### 动态规划

```cpp
class Solution {
public:    
    vector<int> smallestSufficientTeam(vector<string>& req_skills, vector<vector<string>>& people) {
        int len = req_skills.size();    
        unordered_map<int,vector<int> > um;
        unordered_map<string,int> skills;
        um.reserve(1<<len);
        for(int i=0;i<len;i++)
            skills[req_skills[i]] = i;
        um[0] = {};
        
        for(int i=0;i<people.size();i++)
        {
            int cur_skill = 0;
            for(int j=0;j<people[i].size();j++)
            {
                cur_skill |= (1<<skills[people[i][j]]);
            }
            
            for(auto &c:um)
            {
                int comb  = c.first|cur_skill;
                if(um.count(comb)==0||um[comb].size() > (1+um[c.first].size()))
                {
                    um[comb] = c.second;
                    um[comb].push_back(i);
                }
            }
        }
        
        return um[(1<<len)-1];
    }
};
```

