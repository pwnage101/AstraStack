Feature: Color

    # Every input in color/ stores the same scene, so each saves as the same reference image. The tolerances, in
    # 8-bit steps, allow for each encoding's precision; 8-bit linear and YUV encodings are least precise on the
    # patches with a channel at 0.

    Scenario Outline: Every image encoding of the scene saves as the reference
        Given I am on tab "Load"
        And I async press "openFile"
        And I choose input file "color/<file>" from "openDialog"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose output file "<file>.png" from "saveDialog"
        Then output "<file>.png" and reference "<reference>" should be equal within "<tolerance>" where flat

        Examples:
            | file                       | reference             | tolerance |
            | tiff32f_rec2020_linear.tif | color_master_sRGB.png | 1         |
            | png16_srgb_linear.png      | color_master_sRGB.png | 1         |
            | png16_untagged.png         | color_master_sRGB.png | 1         |
            | png16_srgb_srgb.png        | color_master_sRGB.png | 1         |
            | jpg_srgb_srgb.jpg          | color_master_sRGB.png | 2         |
            | jpg_p3_srgb.jpg            | color_master_sRGB.png | 4         |
            | jpg_adobergb_gamma22.jpg   | color_master_sRGB.png | 2         |

    Scenario Outline: Every video encoding of the scene stacks to the reference
        Given I am on tab "Load"
        And I async press "openFile"
        And I choose input file "color/<file>" from "openDialog"
        And I wait until active tab is "Align"
        And I select "None" from "transformation"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose output file "<file>.png" from "saveDialog"
        Then output "<file>.png" and reference "<reference>" should be equal within "<tolerance>" where flat

        Examples:
            | file                         | reference                  | tolerance |
            | mp4_none_none_full.mp4       | color_master_sRGB.png      | 11        |
            | mp4_none_none_none.mp4       | color_master_sRGB.png      | 15        |
            | mp4_bt709_bt709_limited.mp4  | color_master_sRGB.png      | 4         |
            | mp4_bt709_none_limited.mp4   | color_master_sRGB.png      | 15        |
            | mp4_bt709_srgb_full.mp4      | color_master_sRGB.png      | 2         |
            | mp4_10_bt2020_hlg_full.mp4   | color_master_sRGB.png      | 3         |
            | mp4_10_bt2020_pq_limited.mp4 | color_master_sRGB.png      | 9         |
            | avi8_rgb.avi                 | color_master_sRGB.png      | 2         |
            | avi8_gray.avi                | color_master_gray_sRGB.png | 1         |

    Scenario: The float master's values above white come back at 50% Value
        Given I am on tab "Load"
        And I async press "openFile"
        And I choose input file "color/tiff32f_rec2020_linear.tif" from "openDialog"
        And I wait until active tab is "Process"
        And I set "valueAdjust" to "50"
        And I wait "1000"
        And I async press "saveButton"
        And I choose output file "tiff_value50.png" from "saveDialog"
        Then output "tiff_value50.png" and reference "color_master_sRGB_value50.png" should be equal within "1" where flat

    Scenario: The HLG video's values above white come back at 50% Value
        Given I am on tab "Load"
        And I async press "openFile"
        And I choose input file "color/mp4_10_bt2020_hlg_full.mp4" from "openDialog"
        And I wait until active tab is "Align"
        And I select "None" from "transformation"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I set "valueAdjust" to "50"
        And I wait "1000"
        And I async press "saveButton"
        And I choose output file "hlg_value50.png" from "saveDialog"
        Then output "hlg_value50.png" and reference "color_master_sRGB_value50.png" should be equal within "6" where flat

    Scenario: RAW is decoded without white balance or auto-brightening
        Given I am on tab "Load"
        And I async press "openFile"
        And I choose input file "raw/*" from "openDialog"
        And I wait until active tab is "Align"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I check "autoBlackLevel"
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
        And I choose input file "color/tiff32f_rec2020_linear.tif" from "openDialog"
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
        And I choose input file "color/tiff32f_rec2020_linear.tif" from "openDialog"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I choose output file "saved.bmp" from "saveDialog"
        Then the error dialog should show "Only .png, .jpg/.jpeg and .tif/.tiff files are supported"
        And I cancel "saveDialog"
        And output "saved.bmp" should not exist

    Scenario: A saved file reloads as sRGB with the same linear values
        Given I am on tab "Load"
        And I async press "openFile"
        And I choose input file "color/tiff32f_rec2020_linear.tif" from "openDialog"
        And I wait until active tab is "Process"
        And I async press "saveButton"
        And I select "16-bit sRGB" from "bitDepth"
        And I choose output file "reload.png" from "saveDialog"
        And I am on tab "Load"
        And I async press "openFile"
        And I choose saved file "reload.png" from "openDialog"
        And I wait until active tab is "Process"
        Then the loaded file should be tagged as sRGB
        And output "reload.png" and input "color/tiff32f_rec2020_linear.tif" should decode to the same linear values
