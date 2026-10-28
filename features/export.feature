Feature: Export

    Scenario: Invalid pixel formats are insensitive
        Given I am on tab "Load"
        And I async press "openImage"
        And I choose input file "test.jpg" from "openDialog"
        And I wait until active tab is "Process"
        And I press "exportButton"
        And I select "PNG" from "fileType"
        Then "pixelFormat8" should be sensitive
        And "pixelFormat16" should be sensitive
        And "pixelFormat16Float" should not be sensitive
        And "pixelFormat32Float" should not be sensitive
        Given I select "JPEG" from "fileType"
        Then "pixelFormat8" should be active
        And "pixelFormat8" should be sensitive
        And "pixelFormat16" should not be sensitive
        And "pixelFormat16Float" should not be sensitive
        And "pixelFormat32Float" should not be sensitive
        Given I select "TIFF" from "fileType"
        Then "pixelFormat8" should be active
        And "pixelFormat8" should be sensitive
        And "pixelFormat16" should be sensitive
        And "pixelFormat16Float" should be sensitive
        And "pixelFormat32Float" should be sensitive
        Given I check "pixelFormat32Float"
        And I select "EXR" from "fileType"
        Then "pixelFormat32Float" should be active
        And "pixelFormat8" should not be sensitive
        And "pixelFormat16" should not be sensitive
        And "pixelFormat16Float" should be sensitive
        And "pixelFormat32Float" should be sensitive
        Given I select "PNG" from "fileType"
        Then "pixelFormat16" should be active
        Given I select "EXR" from "fileType"
        Then "pixelFormat16Float" should be active

    Scenario: Mismatched extension is replaced
        Given I am on tab "Load"
        And I async press "openImage"
        And I choose input file "test.jpg" from "openDialog"
        And I wait until active tab is "Process"
        And I set neutral levels
        And I press "exportButton"
        Then "exportTab" should be sensitive
        Given I select "EXR" from "fileType"
        And I async press "saveButton"
        And I choose output file "mismatch.png" from "saveDialog"
        Then output "mismatch.exr" should be a "EXR" "16-bit float" export of input "test.jpg"
        And output "mismatch.png" should not exist

    Scenario: EXR carries the input's ICC profile
        Given I generate the Display P3 PNG output file "p3_icc.png"
        And I generate the untagged PNG output file "untagged_icc.png"
        And I am on tab "Load"
        And I async press "openImage"
        And I choose output file "p3_icc.png" from "openDialog"
        And I wait until active tab is "Process"
        And I wait until processing is finished
        And I export output file "p3_icc.exr" as "EXR" "32-bit float"
        Then output "p3_icc.exr" should carry the ICC profile of output "p3_icc.png"
        Given I am on tab "Load"
        And I async press "openImage"
        And I choose output file "untagged_icc.png" from "openDialog"
        And I wait until active tab is "Process"
        And I wait until processing is finished
        And I export output file "untagged_icc.exr" as "EXR" "32-bit float"
        Then output "untagged_icc.exr" should be tagged linear "sRGB"
        And output "untagged_icc.exr" should not carry an ICC profile
