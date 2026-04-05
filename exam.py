from typing import List
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                return mid + 1
            else:
                return mid - 1
        

if __name__ == "__main__":
    solution = Solution()
    nums = [1, 3, 5, 6, 7]
    print(solution.searchInsert(nums=nums, target=7))  # Output: 2
    # print(nums)  # Output: [1, 2, 2]