
# 638. Shopping Offers
[TOC]

[题目链接](https://leetcode.com/problems/shopping-offers/)

### 思路
* 用回溯法，每一步的决策有这几种：
    * 不用offer，那么直接买，计算花费
    * 考虑买每一种offer，选择花费最小的继续递归
* 记得改变了全局变量之后要变回来

### 代码

#### dfs O(N^2)
```cpp
class Solution {
public:
    int shoppingOffers(vector<int>& price, vector<vector<int>>& special, vector<int>& needs,int cost = 0) {
        for(auto n:needs)
        {
            if(n<0) return INT_MAX;
        }
        
        //直接买，
        int m = inner_product(price.begin(),price.end(),needs.begin(),cost);
        
        for(int i=0;i<special.size();i++)
        {
             if(special[i][needs.size()] + cost >= m) continue;
             
            for(int k=0;k<needs.size();k++)
            {
                needs[k] -= special[i][k];
            }
            
            m = min(m,shoppingOffers(price,special,needs,cost+special[i][needs.size()]));

            for(int k=0;k<needs.size();k++)
            {
                needs[k] += special[i][k];
            }
        }
        
        return m;
    }
};
```

