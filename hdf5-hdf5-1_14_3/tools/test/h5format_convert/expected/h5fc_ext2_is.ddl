HDF5 "h5fc_ext2_is-tmp.h5" {
SUPER_BLOCK {
   SUPERBLOCK_VERSION 2
   FREELIST_VERSION 0
   SYMBOLTABLE_VERSION 0
   OBJECTHEADER_VERSION 0
   OFFSET_SIZE 8
   LENGTH_SIZE 8
   BTREE_RANK 16
   BTREE_LEAF 4
   ISTORE_K 64
   FILE_SPACE_STRATEGY H5F_FSPACE_STRATEGY_FSM_AGGR
   FREE_SPACE_PERSIST FALSE
   FREE_SPACE_SECTION_THRESHOLD 1
   FILE_SPACE_PAGE_SIZE 4096
   USER_BLOCK {
      USERBLOCK_SIZE 0
   }
}
GROUP "/" {
   DATASET "DSET_CONTIGUOUS" {
      DATATYPE  H5T_STD_I32LE
      DATASPACE  SIMPLE { ( 10 ) / ( 10 ) }
   }
   DATASET "DSET_EA" {
      DATATYPE  H5T_STD_I32LE
      DATASPACE  SIMPLE { ( 4, 6 ) / ( 10, H5S_UNLIMITED ) }
   }
   DATASET "DSET_FA" {
      DATATYPE  H5T_STD_I32LE
      DATASPACE  SIMPLE { ( 4, 6 ) / ( 20, 10 ) }
   }
   DATASET "DSET_NDATA_BT2" {
      DATATYPE  H5T_STD_I32LE
      DATASPACE  SIMPLE { ( 4, 6 ) / ( H5S_UNLIMITED, H5S_UNLIMITED ) }
   }
   DATASET "DSET_NONE" {
      DATATYPE  H5T_STD_I32LE
      DATASPACE  SIMPLE { ( 4, 6 ) / ( 4, 6 ) }
   }
   GROUP "GROUP" {
      DATASET "DSET_BT2" {
         DATATYPE  H5T_STD_I32LE
         DATASPACE  SIMPLE { ( 4, 6 ) / ( H5S_UNLIMITED, H5S_UNLIMITED ) }
      }
      DATASET "DSET_NDATA_EA" {
         DATATYPE  H5T_STD_I32LE
         DATASPACE  SIMPLE { ( 4, 6 ) / ( 10, H5S_UNLIMITED ) }
      }
      DATASET "DSET_NDATA_FA" {
         DATATYPE  H5T_STD_I32LE
         DATASPACE  SIMPLE { ( 4, 6 ) / ( 20, 10 ) }
      }
      DATASET "DSET_NDATA_NONE" {
         DATATYPE  H5T_STD_I32LE
         DATASPACE  SIMPLE { ( 4, 6 ) / ( 4, 6 ) }
      }
   }
}
}