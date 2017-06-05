# conceptual\_algebras
Some simple functions to explore conceptual algebras



## FILE conceptual\_algebras.py 

Simple functions and classes to study conceptual algebras. See in-code documentation for details. 

## FILE profile\_001.npz 

Profile file for random conceptual algebra, size 10,000, density 0.001
  * To load and visualize:
  ```python 
  >> import conceptual_algebras
  >> import matplotlib.pyplot as plt
  >> p001 = conceptual_algebras.BestSizeProfile(filename="profile_001.npz")
  >> p001.plot_phase_tr()
  >> p001.plot_Cs()
  >> p001.plot_Ls()
  >> plt.show()
  ```
  * To generate (will take a while): 
  ```python
  >> import conceptual_algebras
  >> C = conceptual_algebras.ConceptualSpace(10000, 0.001)
  >> p001 = C.best_size()
  ```
  * To save to file (will overwrite old data unless filename is changed!) 
  ```python
  >> p001.save_data("profile_001.npz")
  ```
## FILE profile\_01.npz 

Profile file for random conceptual algebra, size 10,000, density 0.01

  * To load and visualize: 
  ```python
  >> import conceptual_algebras
  >> import matplotlib.pyplot as plt
  >> p01 = conceptual_algebras.BestSizeProfile(filename="profile_01.npz")
  >> p01.plot_phase_tr()
  >> p01.plot_Cs()
  >> p01.plot_Ls()
  >> plt.show()
  ```
  * To generate (will take a while): 
  ```python
  >> import conceptual_algebras
  >> C = conceptual_algebras.ConceptualSpace(10000, 0.01)
  >> p01 = C.best_size()
  ```
  * To save to file (will overwrite old data unless filename is changed!) 
  ```python
  >> p001.save_data("profile_01.npz")
  ```
