###############################################################################
# - Find LIBRPA
#
find_path(RPA_DIR
    librpa.h
    HINTS ${LIBRPA_DIR}
)
find_library(RPA_LIBRARY
    NAMES rpa
    HINTS ${LIBRPA_LIB_DIR}
    )


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RPA DEFAULT_MSG RPA_LIBRARY RPA_DIR)

if(RPA_FOUND)
    set(RPA_LIBRARY ${RPA_LIBRARY})
    set(RPA_DIR ${RPA_DIR})
    if(NOT TARGET RPA::RPA)
        add_library(RPA::RPA UNKNOWN IMPORTED)
        message(STATUS "RPA_LIBRARY: ${RPA_LIBRARY}")
        message(STATUS "RPA_DIR: ${RPA_DIR}")
        set_target_properties(RPA::RPA PROPERTIES
           IMPORTED_LINK_INTERFACE_LANGUAGES "C"
           IMPORTED_LOCATION "${RPA_LIBRARY}"
           INTERFACE_INCLUDE_DIRECTORIES "${RPA_DIR}")
    endif()
endif()

set(CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES} ${RPA_DIR})
mark_as_advanced(RPA_DIR RPA_LIBRARY)
