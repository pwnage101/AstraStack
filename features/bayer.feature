Feature: Bayer Mosaic

    # Every frame is stacked uncropped, so the outputs keep the input size and depend little on frame selection and
    # alignment. The same scene in each format stacks to about the same intensity with the same settings; the DNG is
    # more saturated, since it is read through its color matrix.
    Scenario Outline: Stacking a Bayer RGGB <kind>
        Given I am on tab "Load"
        And I async press "openFile"
        And I choose input file "<input>" from "openDialog"
        And I wait until active tab is "Align"
        And I select "<colorMode>" from "colorMode"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I uncheck "autoCrop"
        And I set "limitPercent" to "100"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I check "autoBlackLevel"
        And I set "valueAdjust" to "400"
        And I set "redAdjust" to "200"
        And I set "blueAdjust" to "150"
        And I async press "saveButton"
        And I choose output file "<output>" from "saveDialog"
        Then output "<output>" and reference "<output>" should be equal

        Examples:
            | kind                 | input                  | colorMode  | output                   |
            | 16-bit DNG sequence  | dng16_rggb_sequence/*  | RGB        | dng16_rggb_sequence.png  |
            | 16-bit FITS sequence | fits16_rggb_sequence/* | Bayer RGGB | fits16_rggb_sequence.png |
            | 8-bit AVI video      | avi8_rggb.avi          | Bayer RGGB | avi8_rggb.png            |
