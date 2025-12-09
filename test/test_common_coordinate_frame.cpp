#include <gtest/gtest.h>

#include "common/coordinate_frame.hpp"

#include <sstream>

using namespace common;

// Test fixture for CoordinateFrame tests
class CoordinateFrameTest : public ::testing::Test {
protected:
    std::stringstream ss;
};

// Test output stream operator
TEST_F(CoordinateFrameTest, OutputStreamECI) {
    ss << CoordinateFrame::ECI;
    EXPECT_EQ(ss.str(), "ECI");
}

TEST_F(CoordinateFrameTest, OutputStreamECEF) {
    ss << CoordinateFrame::ECEF;
    EXPECT_EQ(ss.str(), "ECEF");
}

// Test input stream operator with valid uppercase inputs
TEST_F(CoordinateFrameTest, InputStreamECIUppercase) {
    CoordinateFrame frame;
    ss << "ECI";
    ss >> frame;
    EXPECT_EQ(frame, CoordinateFrame::ECI);
}

TEST_F(CoordinateFrameTest, InputStreamECEFUppercase) {
    CoordinateFrame frame;
    ss << "ECEF";
    ss >> frame;
    EXPECT_EQ(frame, CoordinateFrame::ECEF);
}

// Test input stream operator with valid lowercase inputs
TEST_F(CoordinateFrameTest, InputStreamECILowercase) {
    CoordinateFrame frame;
    ss << "eci";
    ss >> frame;
    EXPECT_EQ(frame, CoordinateFrame::ECI);
}

TEST_F(CoordinateFrameTest, InputStreamECEFLowercase) {
    CoordinateFrame frame;
    ss << "ecef";
    ss >> frame;
    EXPECT_EQ(frame, CoordinateFrame::ECEF);
}

// Test input stream operator with invalid inputs
TEST_F(CoordinateFrameTest, InputStreamInvalidThrows) {
    CoordinateFrame frame;
    ss << "INVALID";
    EXPECT_THROW(ss >> frame, std::invalid_argument);
}

TEST_F(CoordinateFrameTest, InputStreamEmptyThrows) {
    CoordinateFrame frame;
    ss << "XYZ";
    EXPECT_THROW(ss >> frame, std::invalid_argument);
}

TEST_F(CoordinateFrameTest, InputStreamMixedCaseThrows) {
    CoordinateFrame frame;
    ss << "Eci";
    EXPECT_THROW(ss >> frame, std::invalid_argument);
}

// Test exception message
TEST_F(CoordinateFrameTest, InvalidInputExceptionMessage) {
    CoordinateFrame frame;
    ss << "BAD";
    try {
        ss >> frame;
        FAIL() << "Expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_STREQ(e.what(), "Invalid coordinate frame: BAD");
    }
}

// Test round-trip (output then input)
TEST_F(CoordinateFrameTest, RoundTripECI) {
    CoordinateFrame original = CoordinateFrame::ECI;
    ss << original;
    
    CoordinateFrame parsed;
    ss >> parsed;
    
    EXPECT_EQ(parsed, original);
}

TEST_F(CoordinateFrameTest, RoundTripECEF) {
    CoordinateFrame original = CoordinateFrame::ECEF;
    ss << original;
    
    CoordinateFrame parsed;
    ss >> parsed;
    
    EXPECT_EQ(parsed, original);
}

// Test equality and inequality
TEST_F(CoordinateFrameTest, Equality) {
    CoordinateFrame frame1 = CoordinateFrame::ECI;
    CoordinateFrame frame2 = CoordinateFrame::ECI;
    EXPECT_EQ(frame1, frame2);
}

TEST_F(CoordinateFrameTest, Inequality) {
    CoordinateFrame frame1 = CoordinateFrame::ECI;
    CoordinateFrame frame2 = CoordinateFrame::ECEF;
    EXPECT_NE(frame1, frame2);
}

// Test multiple reads from same stream
TEST_F(CoordinateFrameTest, MultipleReads) {
    ss << "ECI ECEF eci ecef";
    
    CoordinateFrame frame1, frame2, frame3, frame4;
    ss >> frame1 >> frame2 >> frame3 >> frame4;
    
    EXPECT_EQ(frame1, CoordinateFrame::ECI);
    EXPECT_EQ(frame2, CoordinateFrame::ECEF);
    EXPECT_EQ(frame3, CoordinateFrame::ECI);
    EXPECT_EQ(frame4, CoordinateFrame::ECEF);
}

// Test assignment
TEST_F(CoordinateFrameTest, Assignment) {
    CoordinateFrame frame1 = CoordinateFrame::ECI;
    CoordinateFrame frame2;
    frame2 = frame1;
    EXPECT_EQ(frame2, CoordinateFrame::ECI);
}

// Test default construction and assignment
TEST_F(CoordinateFrameTest, DefaultConstruction) {
    CoordinateFrame frame = CoordinateFrame::ECI;
    EXPECT_EQ(frame, CoordinateFrame::ECI);
    
    frame = CoordinateFrame::ECEF;
    EXPECT_EQ(frame, CoordinateFrame::ECEF);
}
