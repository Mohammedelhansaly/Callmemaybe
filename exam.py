from typing import List
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        res = 0
        for i in range(len(digits)):
            res += digits[i] * (10 ** (len(digits) - 1 - i))
        res += 1
        return [int(x) for x in str(res)]


        

if __name__ == "__main__":
    solution = Solution()
    nums = [1, 3, 5, 6, 7]
    print(solution.plusOne([9]))  # Output: 2
    # print(nums)  # Output: [1, 2, 2]