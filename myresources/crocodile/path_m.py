

# from typing import Optional, Generic, List as ListType, TypeVar


#     def __getitem__(self, key: str or list or slice) ->:
#         if isinstance(key, list): return List(self[item] for item in key)  # to allow fancy indexing like List[1, 5, 6]
#         elif isinstance(key, str): return List(item[key] for item in self.list)  # access keys like dictionaries.
#         return self.list[key] if not isinstance(key, slice) else List(self.list[key])  # must be an integer or slice: behaves similarly to Numpy A[1] vs A[1:2]


# a = List[int]([1, 2, 3])[1]
