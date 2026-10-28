Feature: Image Sequence

    Scenario: Stacking Image Sequence
        Given I am on tab "Load"
        And I async press "openSequence"
        And I choose input file "sequence/*" from "openDialog"
        And I wait until active tab is "Align"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I set "limitPercent" to "100"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I press "exportButton"
        And I select "PNG" from "fileType"
        And I check "pixelFormat8"
        And I async press "saveButton"
        And I choose output file "seq.png" from "saveDialog"
        And I check "pixelFormat16"
        And I async press "saveButton"
        And I choose output file "seq_16bit.png" from "saveDialog"
        Then output "seq.png" and reference "seq.png" should be equal
        And output "seq_16bit.png" and reference "seq_16bit.png" should be equal
        
    Scenario: Testing Align RGB
        Given I am on tab "Load"
        And I async press "openSequence"
        And I choose input file "CA.png" from "openDialog"
        And I wait until active tab is "Align"
        And I select "None" from "transformation"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I check "alignChannels"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I press "exportButton"
        And I select "PNG" from "fileType"
        And I check "pixelFormat8"
        And I async press "saveButton"
        And I choose output file "CA.png" from "saveDialog"
        Then output "CA.png" and reference "CA.png" should be equal
        
    Scenario: Stacked image is not clipped
        Given I am on tab "Load"
        And I async press "openSequence"
        And I choose input file "dso.jpg" from "openDialog"
        And I wait until active tab is "Align"
        And I select "None" from "transformation"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I select "2.0X" from "drizzleFactor"
        And I select "Lanczos" from "drizzleInterpolation"
        And I press "stackButton"
        And I wait until active tab is "Process"
        Then the stacked image should have values above 1
