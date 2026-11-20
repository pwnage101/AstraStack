Feature: Bayer Mosaic

    # Every frame is stacked uncropped, so the outputs keep the input size and depend little on frame selection and
    # alignment. The DNG is read through its camera matrix and states its own black level, so it has its own
    # reference. The FITS and AVI state no primaries, so they show camera RGB as sRGB, less saturated, and share a
    # reference built from the FITS output; their auto black level finds the black their files only give in comments.
    # The tolerances, in 8-bit steps, allow for each encoding's precision, the 8-bit AVI's being the largest. The DNG and
    # FITS leave the demosaic choices on Auto (the FITS states its pattern); the AVI picks On and RGGB.
    Scenario Outline: Stacking a Bayer RGGB <kind>
        Given I am on tab "Load"
        And I async press "openFile"
        And I choose input file "<input>" from "openDialog"
        And I wait until active tab is "Align"
        And I am on tab "Load"
        And I select "<demosaic>" from "demosaicMode"
        And I select "<pattern>" from "mosaicPattern"
        And I select "<algorithm>" from "demosaicAlgorithm"
        And I am on tab "Align"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I uncheck "autoCrop"
        And I set "limitPercent" to "100"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I set "autoBlackLevel" checkbox to "<autoBlack>"
        And I set "valueAdjust" to "400"
        And I set "redAdjust" to "<red>"
        And I set "blueAdjust" to "<blue>"
        And I async press "saveButton"
        And I choose output file "<output>" from "saveDialog"
        Then output "<output>" and reference "<reference>" should be equal within "<tolerance>"

        Examples:
            | kind                 | input                  | demosaic | pattern | algorithm | reference             | autoBlack | red | blue | tolerance | output                   |
            | 16-bit DNG sequence  | dng16_rggb_sequence/*  | Auto     | Auto    | Auto      | bayer_rggb_dng.png    | off       | 70  | 100  | 0         | dng16_rggb_sequence.png  |
            | 16-bit FITS sequence | fits16_rggb_sequence/* | Auto     | Auto    | Auto      | bayer_rggb_camera.png | on        | 175 | 38   | 0         | fits16_rggb_sequence.png |
            | 8-bit AVI video      | avi8_rggb.avi          | On       | RGGB    | Auto      | bayer_rggb_camera.png | on        | 175 | 38   | 31        | avi8_rggb.png            |
