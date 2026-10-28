Feature: Process

    Scenario: Black Level and Exposure are applied
        Given I am on tab "Load"
        And I async press "openImage"
        And I choose input file "test.jpg" from "openDialog"
        And I wait until active tab is "Process"
        And I wait until processing is finished
        Then the levels should be automatic
        And "blackLevel" should not be sensitive
        And "exposure" should not be sensitive
        Given I remember the levels
        And I uncheck "autoBlackLevel"
        And I uncheck "autoExposure"
        Then "blackLevel" should be sensitive
        And "exposure" should be sensitive
        And the levels should be the remembered ones
        Given I set "blackLevelAdjust" to "0"
        And I set "exposureAdjust" to "1"
        And I wait until processing is finished
        Then the image should have black level "0.01" and exposure "1"
        Given I set "blackLevelAdjust" to "-2"
        And I set "exposureAdjust" to "-0.5"
        And I wait until processing is finished
        Then the image should have black level "0" and exposure "-0.5"
        Given I check "autoBlackLevel"
        And I check "autoExposure"
        And I wait until processing is finished
        Then the levels should be automatic

    Scenario: Automatic levels are recomputed for a new image
        Given I am on tab "Load"
        And I async press "openImage"
        And I choose input file "test.jpg" from "openDialog"
        And I wait until active tab is "Process"
        And I wait until processing is finished
        And I remember the levels
        And I am on tab "Load"
        And I async press "openImage"
        And I choose input file "dso.jpg" from "openDialog"
        And I wait until active tab is "Process"
        And I wait until processing is finished
        Then the levels should be automatic
        And the levels should be different from the remembered ones

    Scenario: Automatic levels are recomputed for a new stack
        Given I am on tab "Load"
        And I async press "openSequence"
        And I choose input file "sequence/*" from "openDialog"
        And I wait until active tab is "Align"
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I set "limitPercent" to "100"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I wait until processing is finished
        Then the levels should be automatic
        And "blackLevel" should not be sensitive
        And "exposure" should not be sensitive
        Given I am on tab "Stack"
        And I set "limitPercent" to "20"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I wait until processing is finished
        Then the levels should be automatic

    Scenario: Non-linear float export notice
        Given I am on tab "Load"
        And I async press "openImage"
        And I choose input file "test.jpg" from "openDialog"
        And I wait until active tab is "Process"
        And I press "exportButton"
        And I select "TIFF" from "fileType"
        And I check "pixelFormat32Float"
        Then "nonLinearNotice" should not be visible
        Given I am on tab "Process"
        And I select "Square Root" from "stretchFunction"
        And I am on tab "Export"
        Then "nonLinearNotice" should be visible
        Given I check "pixelFormat16"
        Then "nonLinearNotice" should not be visible
        Given I check "pixelFormat16Float"
        Then "nonLinearNotice" should be visible
        Given I am on tab "Process"
        And I select "None" from "stretchFunction"
        And I set "gammaAdjust" to "120"
        And I am on tab "Export"
        Then "nonLinearNotice" should be visible
        Given I select "PNG" from "fileType"
        Then "nonLinearNotice" should not be visible
        Given I select "EXR" from "fileType"
        Then "nonLinearNotice" should be visible
        Given I am on tab "Process"
        And I set "gammaAdjust" to "100"
        And I am on tab "Export"
        Then "nonLinearNotice" should not be visible
