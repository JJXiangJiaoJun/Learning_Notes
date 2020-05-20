# 1074. Number of Submatrices That Sum to Target
[题目链接](https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/)

### 思路
* 这题和**和为K的子数组很类似**，不同的是这题为2D情况
* 考虑一个数组`[[1,2,3,4,5],[2,6,7,8,9]]`,我们可以将其分为一个2x5、以及两个1x5的矩阵
    *  我们从行开始
    *  第一个 1x5的矩阵是[1,2,3,4,5].按照1D方法算即可
    *  我们考虑2x5的矩阵，只需将将其加在上一个矩阵转换成1x5的1D矩阵即可，[3,8,10,12,14].
    *  然后我们考虑从第2行开始
    *  第2行是最后一行，所以我们考虑了所有情况
*  我们可以先预处理出presum，这样就可以在`O(1)`的时间进行查询

### 代码


#### presum+map优化

```cpp
class Solution {
public:
    int numSubmatrixSumTarget(vector<vector<int>>& matrix, int target) {
        if(matrix.size()==0||matrix[0].size()==0) return 0;
        int max_r = matrix.size();
        int max_c = matrix[0].size();
        vector<vector<int> >  presum(max_r+1,vector<int>(max_c,0));
        
        for(int j=0;j<max_c;j++)
            for(int i=0;i<max_r;i++)
            {
                presum[i+1][j] = presum[i][j] + matrix[i][j];  
            }
        
        int ans = 0;
        
        for(int i=0;i<max_r;i++)
        {
            for(int j=i;j<max_r;j++)
            {
                unordered_map<int,int>  um;
                int sum=0;
                //为了计算那些从0开始的subarray,sum是从0开始的
                um[0] = 1;
                for(int k=0;k<max_c;k++)
                {
                    sum += presum[j+1][k]-presum[i][k];
                    if(um.count(sum-target))
                    {
                        ans+=um[sum-target];
                    }
                    um[sum] = um.count(sum)==0?1:um[sum]+1;
                }
                
            }
        }
        return ans;
    }
};
```