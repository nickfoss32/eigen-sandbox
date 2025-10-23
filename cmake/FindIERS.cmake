# cmake/FindIERS.cmake
# Module to download and configure IERS EOP data

# Prevent multiple inclusions
if(IERS_FOUND)
    return()
endif()

# Define option to enable/disable EOP download
option(DOWNLOAD_IERS_EOP "Download IERS EOP data at configure time" ON)

# Define the output directory and file
set(IERS_EOP_DIR "${CMAKE_BINARY_DIR}/data")
set(IERS_EOP_FILE "${IERS_EOP_DIR}/finals2000A.all")
file(MAKE_DIRECTORY "${IERS_EOP_DIR}")

# Download the latest EOP data if enabled
if(DOWNLOAD_IERS_EOP)
    message(STATUS "Attempting to download IERS EOP data to ${IERS_EOP_FILE}")
    file(DOWNLOAD
        "https://maia.usno.navy.mil/ser7/finals2000A.all"  # USNO mirror, updates daily
        "${IERS_EOP_FILE}"
        STATUS IERS_DOWNLOAD_STATUS
        LOG IERS_DOWNLOAD_LOG
        SHOW_PROGRESS
        TLS_VERIFY ON
        TIMEOUT 30
    )

    # Check download status
    list(GET IERS_DOWNLOAD_STATUS 0 IERS_RESULT)
    if(NOT IERS_RESULT EQUAL 0)
        list(GET IERS_DOWNLOAD_STATUS 1 IERS_ERROR_MSG)
        message(FATAL_ERROR "Failed to download IERS EOP data: ${IERS_ERROR_MSG}\nLog: ${IERS_DOWNLOAD_LOG}")
    endif()

    message(STATUS "IERS EOP data downloaded successfully to: ${IERS_EOP_FILE}")
else()
    message(WARNING "IERS EOP download disabled. Ensure ${IERS_EOP_FILE} exists or set DOWNLOAD_IERS_EOP=ON.")
    if(NOT EXISTS "${IERS_EOP_FILE}")
        message(FATAL_ERROR "IERS EOP file not found at ${IERS_EOP_FILE} and DOWNLOAD_IERS_EOP is OFF")
    endif()
endif()

# Set variables for use in the project
set(IERS_EOP_FILE "${IERS_EOP_FILE}" CACHE FILEPATH "Path to downloaded IERS EOP data file")
set(IERS_FOUND TRUE CACHE INTERNAL "IERS EOP data found")
