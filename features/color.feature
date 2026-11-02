Feature: Color

    Scenario Outline: Tagged encodings of the same linear data give the same output
        Given I am on tab "Load"
        And I async press "openFile"
        And I choose input file "color/linear.png" from "openDialog"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose output file "linear_<name>.png" from "saveDialog"
        And I am on tab "Load"
        And I async press "openFile"
        And I choose input file "color/<file>" from "openDialog"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose output file "<name>.png" from "saveDialog"
        Then output "<name>.png" and output "linear_<name>.png" should be equal within "1"

        Examples:
            | file                      | name       |
            | srgb_chunk.png            | srgb_chunk |
            | gama45455.png             | gama45455  |
            | cicp.png                  | cicp       |
            | gama100000.png            | gama100000 |

    Scenario Outline: Tagged videos load and stack
        Given I am on tab "Load"
        And I async press "openFile"
        And I choose input file "color/<file>.mp4" from "openDialog"
        And I wait until active tab is "Align"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose output file "<file>.png" from "saveDialog"
        Then output "<file>.png" and reference "<file>.png" should be equal

        Examples:
            | file  |
            | bt709 |
            | bt601 |
            | srgb  |

    Scenario: Untagged YUV video uses the BT.709 matrix and no curve
        Given I am on tab "Load"
        And I async press "openFile"
        And I choose input file "Jupiter.mp4" from "openDialog"
        And I wait until active tab is "Align"
        And I set "endFrame" to "5"
        And I select "None" from "transformation"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose output file "untagged.png" from "saveDialog"
        And I am on tab "Load"
        And I async press "openFile"
        And I choose input file "color/bt709_unspecified_trc.mp4" from "openDialog"
        And I wait until active tab is "Align"
        And I select "None" from "transformation"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose output file "unspecified_trc.png" from "saveDialog"
        Then output "unspecified_trc.png" and output "untagged.png" should be equal within "1"

    Scenario: Limited range video is expanded on load
        Given I am on tab "Load"
        And I async press "openFile"
        And I choose input file "color/full_range.mp4" from "openDialog"
        And I wait until active tab is "Align"
        And I select "None" from "transformation"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose output file "full_range.png" from "saveDialog"
        And I am on tab "Load"
        And I async press "openFile"
        And I choose input file "color/limited_range.mp4" from "openDialog"
        And I wait until active tab is "Align"
        And I select "None" from "transformation"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose output file "limited_range.png" from "saveDialog"
        Then output "limited_range.png" and output "full_range.png" should be equal within "2" on average

    Scenario Outline: Untagged RGB and gray video match an untagged PNG sequence
        Given I am on tab "Load"
        And I async press "openFile"
        And I choose input file "color/<kind>.avi" from "openDialog"
        And I wait until active tab is "Align"
        And I select "None" from "transformation"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose output file "<kind>_video.png" from "saveDialog"
        And I am on tab "Load"
        And I async press "openFile"
        And I choose input file "color/<kind>_sequence/*" from "openDialog"
        And I wait until active tab is "Align"
        And I select "None" from "transformation"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose output file "<kind>_sequence.png" from "saveDialog"
        Then output "<kind>_video.png" and output "<kind>_sequence.png" should be equal within "1"
        And output "<kind>_video.png" should be tagged as sRGB
        And output "<kind>_sequence.png" should be tagged as sRGB

        Examples:
            | kind |
            | rgb  |
            | gray |

    Scenario: A saturated Display P3 color clips to sRGB
        Given I am on tab "Load"
        And I async press "openFile"
        And I choose input file "color/p3.jpg" from "openDialog"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose output file "p3.png" from "saveDialog"
        And I am on tab "Load"
        And I async press "openFile"
        And I choose input file "color/linear.png" from "openDialog"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose output file "p3_untagged.png" from "saveDialog"
        Then output "p3.png" at "12,12" should be about "0,207,0" within "12"
        And output "p3.png" at "32,12" should be about "222,0,0" within "12"
        And output "p3.png" and output "p3_untagged.png" should differ

    Scenario: RAW is decoded without white balance or auto-brightening
        Given I am on tab "Load"
        And I async press "openFile"
        And I choose input file "raw/*" from "openDialog"
        And I wait until active tab is "Align"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose output file "raw.png" from "saveDialog"
        And I set "redAdjust" to "195"
        And I set "blueAdjust" to "169"
        And I wait "1000"
        And I async press "saveButton"
        And I choose output file "raw_balanced.png" from "saveDialog"
        Then output "raw.png" in region "230,262,310,328" should have the camera ratios "0.512,1,0.591" of "raw/pixel_crop_1.dng" within "0.1"
        And output "raw_balanced.png" in region "230,262,310,328" should be neutral within "0.1"

    Scenario: A Bayer FITS uses its BAYERPAT and keeps its black offset
        Given I am on tab "Load"
        And I async press "openFile"
        And I choose input file "fits/bayer_1.fits" from "openDialog"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose output file "bayer.png" from "saveDialog"
        Then "guessedColorMode" should equal "5"
        And output "bayer.png" at "0,0" should be about "50,50,50" within "8"

    Scenario Outline: Saved files are tagged as sRGB
        Given I am on tab "Load"
        And I async press "openFile"
        And I choose input file "color/linear.png" from "openDialog"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I select "<bits>-bit sRGB" from "bitDepth"
        And I choose output file "<file>" from "saveDialog"
        Then output "<file>" should be tagged as sRGB
        And output "<file>" should have "<bits>" bits

        Examples:
            | file           | bits |
            | saved.png      | 8    |
            | saved16.png    | 16   |
            | saved.jpg      | 8    |
            | saved.tif      | 8    |
            | saved16.tiff   | 16   |

    Scenario: Saving with an unsupported extension is refused
        Given I am on tab "Load"
        And I async press "openFile"
        And I choose input file "color/linear.png" from "openDialog"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose output file "saved.bmp" from "saveDialog"
        Then the error dialog should show "Only .png, .jpg/.jpeg and .tif/.tiff files are supported"
        And I cancel "saveDialog"
        And output "saved.bmp" should not exist

    Scenario: A saved file reloads as sRGB with the same linear values
        Given I am on tab "Load"
        And I async press "openFile"
        And I choose input file "color/linear.png" from "openDialog"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I select "16-bit sRGB" from "bitDepth"
        And I choose output file "reload.png" from "saveDialog"
        And I am on tab "Load"
        And I async press "openFile"
        And I choose saved file "reload.png" from "openDialog"
        And I wait until active tab is "Process"
        Then the loaded file should be tagged as sRGB
        And output "reload.png" and input "color/linear.png" should decode to the same linear values

    Scenario: An image sequence with mismatched color tags warns
        Given I am on tab "Load"
        And I async press "openFile"
        And I choose input file "color/mismatched_sequence/*" from "openDialog"
        Then the error dialog should show "Not all images have the same color tags"
