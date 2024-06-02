#include "Auxiliary.h"

// Function to check if all elements in an array of atomic bool are IDLE
bool _allIdle(_Atomic bool arr[], int size) {
  bool value;
  for (int i = 0; i < size; i++) {
    value = atomic_load(&arr[i]);
    if (value == false) {
      return false;
    }
  }
  return true;
}

// Function to check if all elements in arr are IDLE and update flag accordingly
bool allIdle(_Atomic bool arr[], int size, _Atomic bool *flag) {
  bool value = atomic_load(flag);
  if (value) {
    return true; // fast exit
  } else {
    if (_allIdle(arr, size)) {
      atomic_store(flag,true);
      return true;
    } else {
      return false;
    }
  }
}

void permute(int* arr, int n) {
  for (int i = 0; i < n; i++) {
    arr[i] = i;
  }
  
  // Iterate over each element in the array
  for (int i = n - 1; i > 0; i--) {
    // Select a random index from 0 to i (inclusive)
    int j = rand() % (i + 1);
    
    // Swap arr[i] with the randomly selected element arr[j]
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
  }
}

// Function to find the minimum value in an array of integers
int findMin(int arr[], int size) {
  int minVal = arr[0];  // Initialize minVal with the first element
  
  // Iterate through the array to find the minimum value
  for (int i = 1; i < size; i++) {
    if (arr[i] < minVal) {
      minVal = arr[i];  // Update minVal if current element is smaller
    }
  }
  
  return minVal;  // Return the minimum value
}
