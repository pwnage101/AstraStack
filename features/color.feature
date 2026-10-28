Feature: Color

    Scenario: Display P3 image is converted to sRGB
        Given I generate the Display P3 PNG output file "p3.png"
        And I am on tab "Load"
        And I async press "openImage"
        And I choose output file "p3.png" from "openDialog"
        And I wait until active tab is "Process"
        Then the color space should be "Display P3 test" with transfer "ICC"
        And the color space primaries should be "Display P3"
        Given I set neutral levels
        And I press "exportButton"
        And I select "PNG" from "fileType"
        And I check "pixelFormat8"
        And I async press "saveButton"
        And I choose output file "p3_export.png" from "saveDialog"
        Then output "p3_export.png" should match output "p3.png" converted to sRGB within "1"

    Scenario: LUT based profile is converted to Rec.2020
        Given I generate the LUT profile PNG output file "lut.png"
        And I generate the untagged PNG output file "untagged.png"
        And I am on tab "Load"
        And I async press "openImage"
        And I choose output file "lut.png" from "openDialog"
        And I wait until active tab is "Process"
        Then the color space should be "ITU-R BT.2020" with transfer "Gamma"
        Given I set neutral levels
        And I press "exportButton"
        And I select "PNG" from "fileType"
        And I check "pixelFormat8"
        And I async press "saveButton"
        And I choose output file "lut_export.png" from "saveDialog"
        # 8-bit Rec.2020 intermediate (Pillow only converts 8-bit), saturated colors lose precision
        Then output "lut_export.png" should match output "untagged.png" within "12"

    Scenario: RAW is decoded linear in camera RGB
        Given I generate the DNG output file "raw.dng"
        And I am on tab "Load"
        And I async press "openImage"
        And I choose output file "raw.dng" from "openDialog"
        And I wait until active tab is "Process"
        Then the color space should be "Camera RGB" with transfer "Linear"
        And the color space white point should be "0.3262,0.3498"
        Given I press "exportButton"
        And I select "PNG" from "fileType"
        And I check "pixelFormat8"
        And I async press "saveButton"
        And I choose output file "raw_export.png" from "saveDialog"
        Then output "raw_export.png" should be neutral at "30,20"

    Scenario: Sequence with different color spaces shows a warning
        Given I generate the untagged PNG output file "mixed/1.png"
        And I generate the Display P3 PNG output file "mixed/2.png"
        And I am on tab "Load"
        And I async press "openSequence"
        And I choose output file "mixed/*" from "openDialog"
        Then the error "Not all images have the same color space.  The color space of the first image is used for all of them." should be shown
        Given I wait until active tab is "Align"
        Then the color space should be "sRGB" with transfer "sRGB"
