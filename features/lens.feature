Feature: Lens correction

    Scenario: Nikon NCTG atom is read
        Given I generate the Nikon video output file "nikon.mov"
        And I am on tab "Load"
        And I async press "openVideo"
        And I choose output file "nikon.mov" from "openDialog"
        And I wait until active tab is "Align"
        And I wait until processing is finished
        Then the lens correction should be "Nikon Z 5", "NIKKOR Z 24-70mm f/2.8 S", "24.0", "2.8"
        And "lensDistortion" should be active
        And "lensTCA" should be active
        And "lensVignetting" should be active

    Scenario: Off-center Nikon crop skips geometry corrections
        Given I generate the off-center Nikon video output file "nikon_offcenter.mov"
        And I am on tab "Load"
        And I async press "openVideo"
        And I choose output file "nikon_offcenter.mov" from "openDialog"
        And I wait until active tab is "Align"
        And I wait until processing is finished
        Then "lensDistortion" should not be sensitive
        And "lensTCA" should not be sensitive
        And "lensVignetting" should be sensitive
        And "lensStatus" should contain "The crop isn't centered on the sensor"

    Scenario: EXIF is read from a JPEG
        Given I generate the lens EXIF JPEG output file "lens.jpg"
        And I am on tab "Load"
        And I async press "openImage"
        And I choose output file "lens.jpg" from "openDialog"
        And I wait until active tab is "Process"
        And I wait until processing is finished
        Then the lens correction should be "Canon EOS 5D Mark II", "Canon EF 24-105mm f/4L IS USM", "24.0", "4.0"
        And the lens correction should change the image

    Scenario: Lens correction entered by hand
        Given I am on tab "Load"
        And I async press "openImage"
        And I choose input file "test.jpg" from "openDialog"
        And I wait until active tab is "Process"
        And I wait until processing is finished
        Then there should be no lens correction
        And "lensStatus" should contain "No lens selected"
        Given I am on tab "Load"
        And I select camera "Canon EOS 5D Mark II" and lens "Canon EF 24-105mm f/4L IS USM"
        And I set "lensFocalAdjust" to "24"
        And I set "lensApertureAdjust" to "4"
        And I check "lensDistortion"
        Then the lens correction should change the image

    Scenario: TCA and Align RGB are mutually exclusive
        Given I generate the Nikon video output file "nikon_tca.mov"
        And I am on tab "Load"
        And I async press "openVideo"
        And I choose output file "nikon_tca.mov" from "openDialog"
        And I wait until active tab is "Align"
        And I wait until processing is finished
        Then "lensTCA" should be active
        And "alignChannels" should not be sensitive by itself
        Given I am on tab "Load"
        And I uncheck "lensTCA"
        Then "alignChannels" should be sensitive by itself
        Given I set "alignChannels" active without changing tab
        Then "lensTCA" should not be sensitive

    Scenario: Changing the lens correction discards stacking results
        Given I generate the Nikon video output file "nikon_stack.mov"
        And I am on tab "Load"
        And I async press "openVideo"
        And I choose output file "nikon_stack.mov" from "openDialog"
        And I wait until active tab is "Align"
        And I wait until processing is finished
        And I press "alignButton"
        And I wait until active tab is "Stack"
        And I press "stackButton"
        And I wait until active tab is "Process"
        And I wait until processing is finished
        And I am on tab "Load"
        And I uncheck "lensVignetting"
        Then "alignTab" should be sensitive
        And "stackTab" should not be sensitive
        And "processTab" should not be sensitive
