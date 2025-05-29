using System.ComponentModel.DataAnnotations;

namespace FAIR.Application.DTOs.Identity
{
    public class BaseModel
    {
        
        public required string Username { get; set; }
        public required string Password { get; set; }
    }

}
