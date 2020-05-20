# 902. Numbers At Most N Given Digit Set

[题目链接](https://leetcode.com/problems/numbers-at-most-n-given-digit-set/)

[TOC]

### 思路

* 第一个循环计算位数比N少的个数，为dsize^1 + dsize^2 + dsize^3 + ...........
* 第二个循环处理 xxxx （和N位数相同）的情况，从左至右,
* 1xxx  dsize^3
* 21xx  dsize^2
* .....
* 如果能够组成N，那么最后答案+1


* **以后要计算一个数的位数可以用to_string()处理** 





#### 动态规划

```cpp
class Solution {
public:
    int atMostNGivenDigitSet(vector<string>& D, int N) {
        string NS = to_string(N);
        int digit = NS.size(),dsize = D.size();
        
        int ans =0 ;
        for(int i = 1;i<digit;i++)
            ans += pow(dsize,i);
        
        for(int i = 0;i<digit;i++)
        {
            bool hasSameNum = false;
            for(string &d:D)
            {
                if(d[0] < NS[i]) 
                    ans += pow(dsize,digit-i-1);
                else if(d[0] == NS[i])
                    hasSameNum = true;
            }
            
            if(!hasSameNum) return ans;
        }
        return ans+1;
    }
};
```

