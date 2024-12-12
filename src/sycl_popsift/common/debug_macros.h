#define POP_FATAL(s)                                                                                                   \
    {                                                                                                                  \
        std::stringstream ss;                                                                                          \
        ss << __FILE__ << ":" << __LINE__ << std::endl << "    " << s;                                                 \
        throw std::runtime_error{ss.str()};                                                                            \
    }


// Not sure if i need the following ones
#define POP_FATAL_FL(s, file, line)                                                                                    \
    {                                                                                                                  \
        std::stringstream ss;                                                                                          \
        ss << file << ":" << line << std::endl << "    " << s << std::endl;                                            \
        throw std::runtime_error{ss.str()};                                                                            \
    }

#define POP_CHECK_NON_NULL(ptr,s) if( ptr == 0 ) { POP_FATAL_FL(s,__FILE__,__LINE__); }

#define POP_CHECK_NON_NULL_FL(ptr,s,file,line) if( ptr == 0 ) { POP_FATAL_FL(s,file,line); }

