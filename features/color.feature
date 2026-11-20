Feature: Color

    # Every input in color/ stores the same scene, so each saves as the same reference image. The tolerances, in
    # 8-bit steps, allow for each encoding's precision; 8-bit linear and YUV encodings are least precise on the
    # patches with a channel at 0. At -2EV (Value at 25%) every patch falls below white, so the inputs that hold
    # values above white save them too. The sensor data of the DNG holds nothing above white, so it stores the scene at
    # -2EV, and its rows set Value to 4 times the others.

    Scenario Outline: Every image encoding of the same scene is faithfully decoded
        Given I am on tab "Load"
        And I async press "openFile"
        And I choose input file "color/<file>" from "openDialog"
        And I wait until active tab is "Process"
        And I set "valueAdjust" to "<value>"
        And I wait "1000"
        And I async press "saveButton"
        And I choose output file "<file>_<value>.png" from "saveDialog"
        Then output "<file>_<value>.png" and reference "<reference>" should be equal within "<tolerance>" in mask "color/mask.png"

        Examples:
            | file                    | value | reference                                 | tolerance |
            | tiff32f_none_none.tif   | 100   | color/master_sRGB_+0EV.png                | 1         |
            | tiff16_none_none.tif    | 100   | color/master_sRGB_+0EV.png                | 1         |
            | tiff8_none_none.tif     | 100   | color/master_sRGB_+0EV.png                | 1         |
            | png16_srgb_linear.png   | 100   | color/master_sRGB_+0EV.png                | 1         |
            | png16_none_none.png     | 100   | color/master_sRGB_+0EV.png                | 1         |
            | png16_srgb_srgb.png     | 100   | color/master_sRGB_+0EV.png                | 1         |
            | png16_bt2020_linear.png | 100   | color/master_sRGB_+0EV.png                | 1         |
            | png8_none_none.png      | 100   | color/master_sRGB_+0EV.png                | 1         |
            | jpg_none_none.jpg       | 100   | color/master_sRGB_+0EV.png                | 2         |
            | dng16_camera_none.dng   | 400   | color/master_sRGB_+0EV.png                | 2         |
            | tiff32f_none_none.tif   | 25    | color/master_sRGB_-2EV.png                | 1         |
            | tiff16_none_none.tif    | 25    | color/master_sRGB_clippedSRGB_-2EV.png    | 1         |
            | tiff8_none_none.tif     | 25    | color/master_sRGB_clippedSRGB_-2EV.png    | 1         |
            | png16_srgb_linear.png   | 25    | color/master_sRGB_clippedSRGB_-2EV.png    | 1         |
            | png16_none_none.png     | 25    | color/master_sRGB_clippedSRGB_-2EV.png    | 1         |
            | png16_srgb_srgb.png     | 25    | color/master_sRGB_clippedSRGB_-2EV.png    | 1         |
            | png16_bt2020_linear.png | 25    | color/master_sRGB_clippedRec2020_-2EV.png | 1         |
            | png8_none_none.png      | 25    | color/master_sRGB_clippedSRGB_-2EV.png    | 1         |
            | jpg_none_none.jpg       | 25    | color/master_sRGB_clippedSRGB_-2EV.png    | 1         |
            | dng16_camera_none.dng   | 100   | color/master_sRGB_-2EV.png                | 1         |

    Scenario Outline: Every video encoding of the same scene is faithfully decoded
        Given I am on tab "Load"
        And I async press "openFile"
        And I choose input file "color/<file>" from "openDialog"
        And I wait until active tab is "Align"
        And I select "None" from "transformation"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I set "valueAdjust" to "<value>"
        And I wait "1000"
        And I async press "saveButton"
        And I choose output file "<file>_<value>.png" from "saveDialog"
        Then output "<file>_<value>.png" and reference "<reference>" should be equal within "<tolerance>" in mask "color/mask.png"

        Examples:
            | file                          | value | reference                               | tolerance |
            | mp4_8_none_none_full.mp4      | 100   | color/master_sRGB_+0EV.png              | 11        |
            | mp4_8_none_none_none.mp4      | 100   | color/master_sRGB_+0EV.png              | 15        |
            | mp4_8_bt709_bt709_limited.mp4 | 100   | color/master_sRGB_+0EV.png              | 4         |
            | mp4_8_bt709_none_limited.mp4  | 100   | color/master_sRGB_+0EV.png              | 15        |
            | mp4_8_bt709_srgb_full.mp4     | 100   | color/master_sRGB_+0EV.png              | 2         |
            | mp4_8_bt709_gamma28_limited.mp4 | 100 | color/master_sRGB_+0EV.png              | 4         |
            | mp4_10_bt2020_hlg_full.mp4    | 100   | color/master_sRGB_+0EV.png              | 3         |
            | mp4_10_bt2020_pq_limited.mp4  | 100   | color/master_sRGB_+0EV.png              | 14        |
            | avi8_rgb.avi                  | 100   | color/master_sRGB_+0EV.png              | 2         |
            | avi8_gray.avi                 | 100   | color/master_gray_sRGB_+0EV.png         | 2         |
            | mp4_8_none_none_full.mp4      | 25    | color/master_sRGB_clippedSRGB_-2EV.png  | 3         |
            | mp4_8_none_none_none.mp4      | 25    | color/master_sRGB_clippedSRGB_-2EV.png  | 4         |
            | mp4_8_bt709_bt709_limited.mp4 | 25    | color/master_sRGB_clippedSRGB_-2EV.png  | 2         |
            | mp4_8_bt709_none_limited.mp4  | 25    | color/master_sRGB_clippedSRGB_-2EV.png  | 4         |
            | mp4_8_bt709_srgb_full.mp4     | 25    | color/master_sRGB_clippedSRGB_-2EV.png  | 2         |
            | mp4_8_bt709_gamma28_limited.mp4 | 25  | color/master_sRGB_clippedSRGB_-2EV.png  | 2         |
            | mp4_10_bt2020_hlg_full.mp4    | 25    | color/master_sRGB_-2EV.png              | 3         |
            | mp4_10_bt2020_pq_limited.mp4  | 25    | color/master_sRGB_-2EV.png              | 4         |
            | avi8_rgb.avi                  | 25    | color/master_sRGB_clippedSRGB_-2EV.png  | 1         |
            | avi8_gray.avi                 | 25    | color/master_gray_sRGB_clipped_-2EV.png | 1         |
