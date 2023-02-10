# Flight-Test-Data-Analytics-Module-01

## Setup

This course assumes you have installed the Anaconda Distribution, available at <https://www.anaconda.com/products/distribution#Downloads>

### Conda environment and Spyder

The conda environment used for the module is available as `environment.yml`

To import the environment and open the Spyder IDE, you have two ways of doing so:

1. Anaconda Navigator
    1. Open Anaconda Navigator
    2. From the left sidebar menu select "Environments"
    3. At the bottom of the window select "Import"
    4. From "Local Drive" browse to the folder Flight-Test-Data-Analytics-Module-01 and select `environment.yml`
    5. Name the environment "ftda_m01"
    6. Once installation is complete, select "ftda_m01" from the available environments
    7. Click the play button, and select "Open in Terminal"
    8. In the terminal, type `spyder` and hit Enter
2. Anaconda Terminal/PowerShell
    1. Open Anaconda Terminal or Anaconda PowerShell
    2. In the terminal, navigate to the folder Flight-Test-Data-Analytics-Module-01
    3. Type `conda env create -f environment.yml` and hit Enter
    4. Agree to changes
    5. Once installation is complete, type `spyder` and hit Enter
