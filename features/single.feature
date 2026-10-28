Feature: Single Image

    Scenario: Test making no changes to single image
        Given I am on tab "Load"
        And I async press "openImage"
        And I choose input file "test.jpg" from "openDialog"
        And I wait until active tab is "Process"
        And I set neutral levels
        And I press "exportButton"
        And I select "PNG" from "fileType"
        And I check "pixelFormat8"
        And I async press "saveButton"
        And I choose output file "test_unchanged.png" from "saveDialog"
        Then output "test_unchanged.png" and input "test.jpg" should be equal
  
    Scenario: Test processing single image
        Given I am on tab "Load"
        And I async press "openImage"
        And I choose input file "test.jpg" from "openDialog"
        And I wait until active tab is "Process"
        And I set "sharpen1" to "0.150"
        And I set "sharpen2" to "0.100"
        And I set "denoise1" to "0.5"
        And I set "deconvolveGaussianDiameter" to "2"
        And I press "exportButton"
        And I select "PNG" from "fileType"
        And I check "pixelFormat8"
        And I async press "saveButton"
        And I choose output file "test.png" from "saveDialog"
        Then output "test.png" and reference "test.png" should be equal
        
    Scenario: Test dering
        Given I am on tab "Load"
        And I async press "openImage"
        And I choose input file "dso.jpg" from "openDialog"
        And I wait until active tab is "Process"
        And I set "sharpen2" to "0.100"
        And I set "deringAdaptiveAdjust" to "24"
        And I set "deringSizeAdjust" to "1"
        And I set "deringBlendAdjust" to "2"
        And I press "exportButton"
        And I select "PNG" from "fileType"
        And I check "pixelFormat8"
        And I async press "saveButton"
        And I choose output file "dso.png" from "saveDialog"
        Then output "dso.png" and reference "dso.png" should be equal
        
    Scenario: Test processing 16bit image
        Given I am on tab "Load"
        And I async press "openImage"
        And I choose input file "16bit.png" from "openDialog"
        And I wait until active tab is "Process"
        And I set "sharpen1" to "0.600"
        And I set "sharpen2" to "0.300"
        And I set "gammaAdjust" to "85"
        And I uncheck "autoExposure"
        And I set "exposureAdjust" to "2.2"
        And I press "exportButton"
        And I select "PNG" from "fileType"
        And I check "pixelFormat8"
        And I async press "saveButton"
        And I choose output file "16bit.png" from "saveDialog"
        And I check "pixelFormat16"
        And I async press "saveButton"
        And I choose output file "16bit_16bit.png" from "saveDialog"
        Then output "16bit.png" and reference "16bit.png" should be equal
        And output "16bit_16bit.png" and reference "16bit_16bit.png" should be equal
        
    Scenario: Test masking stretched image
        Given I am on tab "Load"
        And I async press "openImage"
        And I choose input file "test.jpg" from "openDialog"
        And I wait until active tab is "Process"
        And I select "ArcSinH" from "stretchFunction"
        And I set "deringAdaptiveAdjust" to "50"
        And I press "exportButton"
        And I select "PNG" from "fileType"
        And I check "pixelFormat8"
        And I async press "saveButton"
        And I choose output file "mask.png" from "saveDialog"
        Then output "mask.png" and reference "mask.png" should be equal
        
    Scenario: Changing color mode reprocesses single image
        Given I am on tab "Load"
        And I async press "openImage"
        And I choose input file "test.jpg" from "openDialog"
        And I wait until active tab is "Process"
        And I am on tab "Load"
        And I select "Grayscale" from "colorMode"
        And I am on tab "Process"
        And I press "exportButton"
        And I select "PNG" from "fileType"
        And I check "pixelFormat8"
        And I async press "saveButton"
        And I choose output file "test_grayscale.png" from "saveDialog"
        Then output "test_grayscale.png" should be grayscale
